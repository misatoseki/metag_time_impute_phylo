import pickle, math
import numpy as np
import typing as ty
import torch
import torch.nn as nn
import torch.nn.init as nn_init

from torch import Tensor

from diff_models import diff_CSDI


class Tokenizer(nn.Module):
    def __init__(
        self, d_numerical: int, categories: ty.Optional[ty.List[int]], d_token: int, bias: bool,
    ) -> None:
        super().__init__()

        d_bias = len(categories)
        category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.d_token = d_token
        self.register_buffer("category_offsets", category_offsets)
        self.category_embeddings = nn.Embedding(sum(categories) + 1, self.d_token)
        nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        self.category_embeddings.weight.requires_grad = False

        self.bias = nn.Parameter(Tensor(d_bias, self.d_token)) if bias else None
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            self.bias.requires_grad = False

    @property
    def n_tokens(self) -> int:
        return len(self.category_offsets)

    def forward(self, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_cat = x_cat.type(torch.int32)
        x_cat_embed = self.category_embeddings(x_cat + self.category_offsets[None]) # [B, time, num_cat, d_token]
        if self.bias is not None:
            x_cat_embed = x_cat_embed + self.bias[None]

        x_cat_embed = x_cat_embed.reshape(x_cat_embed.shape[0], x_cat_embed.shape[1], -1) # [B, time, num_cat*d_token]

        return x_cat_embed


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, cluster_size):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.cluster_size = cluster_size

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        ### Added
        with open("./data/transformed_columns.pk", "rb") as f:
            cont_list, num_cate_list = pickle.load(f)

        self.cont_list = cont_list

        self.token_dim = config["model"]["token_emb_dim"]

        d_numerical = len(cont_list)
        categories = num_cate_list
        d_token = self.token_dim
        token_bias = True

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        ###

        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2 + len(num_cate_list) * d_token
        self.diffmodel = diff_CSDI(config_diff, input_dim, self.cluster_size)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape # K is error

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B,L,K,emb)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, observed_data_cat
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, observed_data_cat, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, observed_data_cat, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask, observed_data_cat)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask, observed_data_cat):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            # reshape category data
            observed_data_cat = observed_data_cat.permute(0, 2, 1) # [B, d_token, L]
            observed_data_cat = observed_data_cat.unsqueeze(-1)    # [B, d_token, L, 1]
            observed_data_cat = observed_data_cat.expand(-1, -1, -1, len(self.cont_list)) # [B, d_token, L, n_taxon]
            observed_data_cat = observed_data_cat.permute(0, 1, 3, 2) # [B, d_token, n_taxon, L]
            # concatenate by channel dimension
            total_input = torch.cat([cond_obs, noisy_target, observed_data_cat], dim=1)  # [B,(2 + num_cat*d_token),K,L]

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, observed_data_cat):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        # reshape category data
        observed_data_cat = observed_data_cat.permute(0, 2, 1) # [B, d_token, L]
        observed_data_cat = observed_data_cat.unsqueeze(-1)    # [B, d_token, L, 1]
        observed_data_cat = observed_data_cat.expand(-1, -1, -1, len(self.cont_list)) # [B, d_token, L, n_taxon]
        observed_data_cat = observed_data_cat.permute(0, 1, 3, 2) # [B, d_token, n_taxon, L]

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    # concatenate by channel dimension
                    diff_input = torch.cat([cond_obs, noisy_target, observed_data_cat], dim=1)  # [B,(2 + num_cat*d_token),K,L]
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            observed_data_cat
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, observed_data_cat)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            observed_data_cat
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples, observed_data_cat)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_diabimmune(CSDI_base):
    def __init__(self, config, device, target_dim, cluster_size, num_cat):
        super(CSDI_diabimmune, self).__init__(target_dim, config, device, cluster_size)
        self.num_cat = num_cat

    def process_data(self, batch):
        # observed_data
        observed_data = batch["observed_data"].to(self.device).float()
        observed_data = observed_data[:, :, :(observed_data.shape[2] - self.num_cat)] # [B, time, taxon]
        
        # category variables are embedded by feature tokenizer
        observed_data_cat = self.tokenizer(
            observed_data[:, :, -self.num_cat:], # [B, time, num_cat]
        ) # [B, time, num_cat*d_token]

        # observed_mask
        observed_mask = ~torch.isnan(observed_data)
        observed_mask = observed_mask.to(self.device).float()

        # observed_tp
        observed_tp = batch["timepoints"].to(self.device).float()

        # gt_mask
        gt_mask = batch["gt_mask"].to(self.device).float()
        gt_mask = gt_mask[:, :, :(gt_mask.shape[2] - self.num_cat)]

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_data_cat
        )



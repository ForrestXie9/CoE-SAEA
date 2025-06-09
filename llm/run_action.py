
def run_action(self, index1, offspring_c):
    sort_index = np.argsort(self.hf)
    self.ghf = self.hf[sort_index[:self.initial_sample_size]]
    self.ghx = self.hx[sort_index[:self.initial_sample_size]]

    if self.NFEs < self.maxFEs:

        if index1 == 1:
            offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax,
                                   self.VRmin)
            self.hx, self.hf, reward_rp, self.NFEs, self.CE, self.gfs, candidate_fit = RBF_pre_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)
            self.update(index1, reward_rp, offspring_c, candidate_fit)

        elif index1 == 2:
            offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
            self.hx, self.hf, reward_gl, self.NFEs, self.CE, self.gfs, candidate_fit = GP_lcb_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)
            self.update(index1, reward_gl, offspring_c, candidate_fit)

        elif index1 == 3:
            self.hx, self.hf, reward_rl, self.NFEs, self.CE, self.gfs, candidate_fit = RBF_ls_arm(self.ghx, self.ghf, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.cxmin, self.cxmax, self.num_arm, self.paras)
            self.update(index1, reward_rl, offspring_c, candidate_fit)

        elif index1 == 4:
            offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
            self.hx, self.hf, reward_ge, self.NFEs, self.CE, self.gfs, candidate_fit = GP_EI_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)
            self.update(index1, reward_ge, offspring_c, candidate_fit)

        elif index1 == 5:
            offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
            self.hx, self.hf, reward_pp, self.NFEs, self.CE, self.gfs, candidate_fit = PRS_pre_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)
            self.update(index1, reward_pp, offspring_c, candidate_fit)

        elif index1 == 6:
            self.hx, self.hf, reward_pl, self.NFEs, self.CE, self.gfs, candidate_fit = PRS_ls_arm(self.ghx, self.ghf, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.cxmin, self.cxmax, self.num_arm, self.paras)
            self.update(index1, reward_pl, offspring_c, candidate_fit)

        elif index1 == 7:
            offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
            self.hx, self.hf, reward_Gi, self.NFEs, self.CE, self.gfs, candidate_fit = KNN_eoi_arm(self.ghx, self.ghf, offspring, self.hx, self.hf ,self.FUN, self.NFEs, self.level, self.CE, self.gfs, self.num_arm, self.paras)
            self.update(index1, reward_Gi, offspring_c, candidate_fit)

        elif index1 == 8:
            offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
            self.hx, self.hf, reward_Go, self.NFEs, self.CE, self.gfs, candidate_fit = KNN_eor_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.level, self.CE, self.gfs, self.num_arm,                                                                                    self.paras)
            self.update(index1, reward_Go, offspring_c, candidate_fit)
    else:
        candidate_fit = None

    return self.NFEs, candidate_fit
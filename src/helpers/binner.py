import numpy as np

class binner2D:
    def __init__(self, r, Rbins, height=1.0):
        """
        r       : 2D array of radial distances
        R_bins  : 1D array of bin edges (monotonically increasing)
        height  : float, cylinder height (line-of-sight length) for 'vol' statistic
        """
        self.r = r
        self.R_bins = Rbins
        self.height = height

    def apply(self, p , statistic='mean', weights = None, method = "histogram"):
        """
        Bin the property p according to radial bins using np.histogram.

        p       : 2D array of property to bin (same shape as r)
        statistic : 'mean', 'sum', 'std', or 'vol'
        """
        if self.r.shape != p.shape:
            raise ValueError("r and p must have the same shape")
        
        p = p if weights is None else p*weights
        if method == "histogram":
            r = self.r.ravel()
            p = p.ravel()
            counts, _ = np.histogram(r, bins=self.R_bins)
            sum_w, _ = np.histogram(r, bins=self.R_bins, weights=p)
        
            if statistic == 'mean':
                prof = sum_w / counts
            elif statistic == 'sum':
                prof = sum_w
            elif statistic == 'std':
                sum_w2, _ = np.histogram(r, bins=self.R_bins, weights=p**2)
                mean = sum_w / counts
                mean2 = sum_w2 / counts
                prof = np.sqrt(mean2 - mean**2)
            elif statistic == 'vol':
                r_outer = self.R_bins[1:]
                r_inner = self.R_bins[:-1]
                shell_vol = np.pi * (r_outer**2 - r_inner**2) * self.height
                prof = sum_w / shell_vol
            else:
                raise ValueError("statistic must be 'mean', 'sum', 'std', or 'vol'")
        elif method == "where":
            prof = np.zeros(len(self.R_bins) - 1)
            for i in range(len(self.R_bins) - 1):
                ri = self.R_bins[i]
                rf = self.R_bins[i+1]
                mask = np.where((r >= ri) & (r < rf))
                counts = np.sum(mask)
                if statistic == 'mean':
                    prof[i] = np.mean(p[mask])
                elif statistic == 'sum':
                    prof[i] = np.sum(p[mask])
                elif statistic == 'std':
                    prof[i] = np.std(p[mask])
                elif statistic == 'vol':
                    shell_vol = np.pi * (r_outer**2 - r_inner**2) * self.height
                    prof[i] = np.sum(p[mask])/shell_vol
                    
        R_centers = 0.5 * (self.R_bins[:-1] + self.R_bins[1:])
        return R_centers, np.nan_to_num(prof), counts
    
class binner3D:
    def __init__(self, r, Rbins):
        """
        r       : 3D array of radial distances
        R_bins  : 1D array of bin edges (monotonically increasing)
        """
        # flatten for histogram
        
        self.r = r
        self.R_bins = Rbins

    def apply(self, p , statistic='mean', weights = None, method = "histogram"):
        """
        Bin the property p according to radial bins using np.histogram or np.where.

        p       : 3D array of the property to bin (same shape as r)
        statistic : 'mean', 'sum', 'std', or 'vol'
        """
        if self.r.shape != p.shape:
            raise ValueError("r and p must have the same shape")
        p = p if weights is None else p*weights
        if method == "histogram":
            r = self.r.ravel()
            p = p.ravel()
            counts, _ = np.histogram(r, bins=self.R_bins)
            sum_w, _ = np.histogram(r, bins=self.R_bins, weights=p)

            if statistic == 'mean':
                prof = sum_w / counts
            elif statistic == 'sum':
                prof = sum_w
            elif statistic == 'std':
                sum_w2, _ = np.histogram(r, bins=self.R_bins, weights=p**2)
                mean = sum_w / counts
                mean2 = sum_w2 / counts
                prof = np.sqrt(mean2 - mean**2)
            elif statistic == 'vol':
                r_outer = self.R_bins[1:]
                r_inner = self.R_bins[:-1]
                V = 4/3 * np.pi * (r_outer**3 - r_inner**3)
                prof = sum_w / V
            else:
                raise ValueError("statistic must be 'mean', 'sum', 'std', or 'vol'")
        elif method == "where":
            prof = np.zeros(len(self.R_bins) - 1)
            r = self.r
            p = p if weights is None else p*weights
            for i in range(len(self.R_bins) - 1):
                ri = self.R_bins[i]
                rf = self.R_bins[i+1]
                mask = np.where((r >= ri) & (r < rf))
                counts = np.sum(mask)
                if statistic == 'mean':
                    prof[i] = np.mean(p[mask])
                elif statistic == 'sum':
                    prof[i] = np.sum(p[mask])
                elif statistic == 'std':
                    prof[i] = np.std(p[mask])
                elif statistic == 'vol':
                    V = 4/3*np.pi*(rf**3 - ri**3)
                    prof[i] = np.sum(p[mask])/V
        R_centers = 0.5 * (self.R_bins[:-1] + self.R_bins[1:])
        return R_centers, np.nan_to_num(prof), counts 
    
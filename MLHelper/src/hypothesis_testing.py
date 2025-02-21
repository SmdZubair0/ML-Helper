import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats

class TestHypothesis():
    """
        This class contains methods to perform various test hypothesis.
    """
    def __init__(self, significance_level = 0.05):
        """
            inputs:
                significance_level : alpha value (default 0.05)
        """
        self.significance_level = significance_level

    def isNormal(self, feature):
        """
            inputs:
                feature : sequence of data containing numeric values
            output:
                True : if feature is normal
                False : if feature is not normal

        """
        if len(feature) > 5000:  # When sample size > 5000 shapiro doesn't work good.
            p_value = stats.kstest(feature, 'norm', args=(np.mean(feature), np.std(feature))).pvalue
        else:
            p_value = stats.shapiro(feature).pvalue
        return p_value > self.significance_level  # if pvalue > significance level then feature is normally distributed
    
    def hasEqualVariance(self, *features, method = "levene"):
        """
            inputs:
                features : sequence of numeric sequences
                method : method used to calculate variance (default levene)
            output:
                True : if all features have equal variance
                False : if features doesn't have equal variance
        """
        if method == "bartlett":  # works good for normal data
            if any(not self.isNormal(f) for f in features):
                raise ValueError("Bartlett's test requires normality.")
            vstats, p_value = stats.bartlett(*features)
        else:
            vstats, p_value = stats.levene(*features)

        return p_value > self.significance_level  # if pvalue > significance level then both features has equal variance
    
    def checkSignificance(self, p_value):
        """
            inputs:
                p_value : p_value received after hypothesis testing
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        if p_value < self.significance_level:  # if pvalue < alpha we reject null hypothesis
            print("Null Hypothesis rejected.\nThere is a significance difference.")
            return True, p_value
        else:
            print("Not enough evidence to reject Null Hypotheses.\nThere is no significance difference.")
            return False, p_value
        
    def OneSampleTtest(self, feature, meu, clt = False):
        """
            inputs:
                feature : Sequence of numeric data (sample)
                meu : Population mean
                clt : to apply CLT or not (default False)
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        if not self.isNormal(feature): # feature should be normal for t-test
            if not clt:  # if don't want to apply clt
                raise ValueError("Sample is not Normally distributed")
            elif len(feature) <= 30:  # if len(feature) > 30 clt can be applied
                raise ValueError("Unable to apply CLT for feature size < 30")
        
        ttest, p_value = stats.ttest_1samp(feature, meu)

        return self.checkSignificance(p_value)
        

    def twoSampleTtest(self, feature1, feature2, clt = False):
        """
            input:
                feature1 : Sequence of numeric values
                feature2 : Another independent sequence of numeric values
                clt : to apply CLT or not (default False)
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        # please ensure that feature1 and feature2 are of same size and similar scale
        if len(feature1) != len(feature2):
            raise ValueError("Samples are of different sizes")
        
        if not self.isNormal(feature1) or not self.isNormal(feature2): # both features should be normal for t-test
            if not clt:  # if don't want to apply clt
                raise ValueError("Samples are not Normally distributed")
            elif len(feature1) <= 30:  # if len(feature) > 30 clt can be applied
                raise ValueError("Unable to apply CLT for feature size < 30")
        
        equal_var = self.hasEqualVariance(feature1, feature2)  # equal_var will be true if both has equal variance

        tstats, p_value = stats.ttest_ind(feature1, feature2, equal_var=equal_var)

        return self.checkSignificance(p_value)
        
    def pairedTtest(self, feature1, feature2, clt = False):
        """
            input:
                feature1 : Sequence of numeric values
                feature2 : Another dependent sequence of numeric values
                clt : to apply CLT or not (default False)
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        if not self.isNormal(feature1) or not self.isNormal(feature2): # both features should be normal for t-test
            if not clt:  # if don't want to apply clt
                raise ValueError("Samples are not Normally distributed")
            elif len(feature1) <= 30:  # if len(feature) > 30 clt can be applied
                raise ValueError("Unable to apply CLT for feature size < 30")
            
        tstat, p_value = stats.ttest_rel(feature1, feature2)

        return self.checkSignificance(p_value)
        
    def ANOVA(self, *features, clt = False):
        """
            input:
                features : Sequence of numeric sequences
                clt : to apply CLT or not (default False)
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        # please ensure that feature1 and feature2 are of same size and similar scale
        n = len(features[0])
        for feature in features:
            if len(feature) != n:
                warnings.warn("Please ensure all the samples are of same length.\nTry downsampling the larger samples.", category=UserWarning)
                break

        for feature in features:
            if not self.isNormal(feature): # features should be normal for t-test
                if not clt:  # if don't want to apply clt
                    raise ValueError("Sample is not Normally distributed")
                elif len(feature) <= 30:  # if len(feature) > 30 clt can be applied
                    raise ValueError("Unable to apply CLT for feature size < 30")
            
        equal_var = self.hasEqualVariance(*features, method="bartlett")  # equal_var will be true if both has equal variance

        tstats, p_value = stats.f_oneway(*features)

        return self.checkSignificance(p_value)
        
    def mannWhitneyUtest(self, feature1, feature2):
        """
            input:
                feature1 : Sequence of numeric values
                feature2 : Another independent sequence of numeric values
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        # when features are not normally distributed

        len1, len2 = len(feature1), len(feature2)

        if abs(len1 - len2) > 0.5 * min(len1, len2):  # features size doesn't make much impact here
            warnings.warn("Warning: Large sample size difference may affect test accuracy!")

        if self.isNormal(feature1) and self.isNormal(feature2):  # if both features are normal it is better to use t-test
            warnings.warn("Features are normal, Consider using t-test instead")
        
        mstat, p_value = stats.mannwhitneyu(feature1, feature2, alternative="two-sided") # return statistic value and pvalue

        return self.checkSignificance(p_value)
        
    def wilcoxonSignedRanktest(self, feature1, feature2):
        """
            input:
                feature1 : Sequence of numeric values
                feature2 : Another independent sequence of numeric values
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        # when features are not normally distributed

        len1, len2 = len(feature1), len(feature2)

        if len1 != len2:
            raise ValueError("Sample sizes are different.\nTry using samples with same sizes")

        if self.isNormal(feature1) and self.isNormal(feature2):  # if both features are normal it is better to use t-test
            warnings.warn("Features are normal, Consider using Paired sample t-test instead")
        
        mstat, p_value = stats.wilcoxon(feature1, feature2, alternative="two-sided") # return statistic value and pvalue

        return self.checkSignificance(p_value)
        
    def kruskalWallis(self, *features):
        """
            input:
                features : Sequence of numeric sequences
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        kstats, p_value = stats.kruskal(*features)

        return self.checkSignificance(p_value)
        
    def chiSquare(self, feature1, feature2):
        """
            input:
                feature1 : Sequence of categorical values
                feature2 : Another independent sequence of categorical values
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        ct = pd.crosstab(feature1, feature2)  # gives a contengency table comparing two features.
        
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(ct)
        if (expected < 5).sum() > 0.2 * expected.size:  # when expected frequencies are elss than 5
            raise ValueError("Some expected frequencies are too small, Fisher's test is preferred.")

        return self.checkSignificance(p_value)
    
    def fisherExacttest(self, feature1, feature2):
        """
            input:
                feature1 : Sequence of categorical values
                feature2 : Another independent sequence of categorical values
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        if len(feature1) >= 5 and len(feature2) >= 5:  # when sample len > 5, better to use chi-Square test
            warnings.warn("Sample lengths are >=5.\nTry using Chi-Square test.")

        ct = pd.crosstab(feature1, feature2)  # contengency table

        ftest, p_value = stats.fisher_exact(ct)
        
        return self.checkSignificance(p_value)
        
    def pearsonCorrelation(self, feature1, feature2):
        """
            input:
                feature1 : Sequence of numeric values
                feature2 : Another independent sequence of numeric values
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        if len(feature1) != len(feature2): # lenghts should be equal to use pearson Correlation
            raise ValueError("Sample lenghts are not equal.\nTry using downsampling methods.")
        
        if np.var(feature1, ddof=1) == 0 or np.var(feature2, ddof=1) == 0: # ddof = 1 is to find variance of sample, and it should not be 0
            raise ValueError("Variance of feature should not be 0.\nTry using Spearman Correlation instead.")
        
        if not self.isNormal(feature1) or not self.isNormal(feature2): # the sequence should be in normal
            raise ValueError("Samples are not normally distributed.")
        
        corr, p_value = stats.pearsonr(feature1, feature2)

        return self.checkSignificance(p_value)
        
    def spearmanCorrelation(self, feature1, feature2):
        """
            input:
                feature1 : Sequence of numeric values
                feature2 : Another independent sequence of numeric values
            output:
                True, p_value : When the null hypothesis is rejected.
                False, p_value : When the null hypothesis is not rejected.
        """
        if len(feature1) != len(feature2):  # the lengths should be equal
            raise ValueError("Sample lenghts are not equal.\nTry using downsampling methods.")
        
        corr, p_value = stats.spearmanr(feature1, feature2)

        return self.checkSignificance(p_value)
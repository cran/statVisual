# v5 created on Feb. 17, 2019
#  (1) when number of variables are too large, the function 'eigen' will be
#      very slow. So we still use 'prcomp', but with modified input dat, 
#      in which NA is replaced by zero
# v4 created on Feb. 15, 2019
#  (1) use 'pairwise.complete.obs' instead of 'na.or.complete'
#      to handle the cases where all probes have at least one NA
#
# v3 created on Feb. 7, 2019
#  (1) add class to the output object so 'factoextra:::get_eig' can
#      handle the output
# v2 created on Jan. 29, 2019
#  (1) change '.scale' to 'scale.'
#  (2) simplify the R code
#
# created on Jan. 29, 2019
#  (1) calculate principal components when data containing missing values

# improved prcomp

# dat - n x p matrix; rows are subjects and columns are variables

iprcomp=function(dat, center=TRUE, scale. = FALSE)
{
  dat0=dat
  dat0[is.na(dat)] = 0
  
  res=prcomp(dat0, center=center, scale.=scale.)

  invisible(res)
}



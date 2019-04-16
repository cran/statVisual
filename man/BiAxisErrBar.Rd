\name{BiAxisErrBar}
\alias{BiAxisErrBar}
\title{
Compare Patterns of Two Outcomes in One Scatter Plot
}
\description{
Compare patterns of two outcomes with different scales across the range of the common predictor using error bar plots. Each bar plot displays mean \eqn{+/-}{+/-} standard error. 
}
\usage{
BiAxisErrBar(dat, 
	     group, 
	     y.left, 
	     y.right, 
	     col.left = "blue", 
	     col.right = "red", 
	     delta = 0.2, 
	     xlab = group, 
	     ylab.left = y.left, 
	     ylab.right = y.right, 
	     title = "Bi-Axis Error Bar Plot",
	     las = 1, 
	     angle = 90, 
	     length = 0.05, 
	     line = 2, 
	     type = "b", 
	     code = 3,
	     legend.position="topright",
	     ...)
}
\arguments{
  \item{dat}{
A data frame. Rows are subjects; Columns are variables describing the subjects.
}
  \item{group}{
character. A categorical variable in \code{data} that indicates the predictor. 
}
  \item{y.left}{
character. The column name of \code{data} that indicates the first outcome variable, the error bar plot of which will be drawn on the left side.
}
  \item{y.right}{
character. The column name of \code{data} that indicates the second outcome variable, the error bar plot of which will be drawn on the right side.
}
  \item{col.left}{
character. The color of the error bar plot for the first outcome variable.
}
  \item{col.right}{
character. The color of the error bar plot for the second outcome variable.
}
  \item{delta}{
numeric. A small number so that the second error bar plot will shift \code{delta} distance from the first error bar plot.
}
  \item{xlab}{
	  character. Label for the x-axis.
}
  \item{ylab.left}{
	  character. Label for the left-side y-axis.
}
  \item{ylab.right}{
	  character. Label for the right-side y-axis.
}
  \item{title}{
character. title of the plot.
}
  \item{las}{
	  integer. indicates the style of axis labels. Please refer to the manual of the R function \code{\link{par}}.
}
  \item{angle}{
	  numeric. Angle from the shaft of the arrow to the edge of the arrow head. Please refer to the manual of the R function \code{\link{mtext}}.
}
  \item{length}{
	  numeric. Length of the edges of the arrow head (in inches). Please refer to the manual of the R function \code{\link{mtext}}.
}
  \item{line}{
	  numeric. indicate the starting line of text on margin. Please refer to the manual of the R function \code{\link{mtext}}.
}
  \item{type}{
character. indicating the type of plotting. Please refer to the manual of the R function \code{\link{type}}.
}
  \item{code}{
integer. determining kind of arrows to be drawn. Please refer to the manual of the R function \code{\link{arrows}}.
}
  \item{legend.position}{
character. determining the position of the legend. Please refer to the R function \code{\link{legend}}.
}
  \item{\dots}{
	  other input parameters for the R function \code{\link{plot}}.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
	None.
}
%\references{
%% ~put references to the literature/web site here ~
%}
\author{
Wenfei Zhang <Wenfei.Zhang@sanofi.com>,
Weiliang Qiu <Weiliang.Qiu@sanofi.com>,
Xuan Lin <Xuan.Lin@sanofi.com>,
Donghui Zhang <Donghui.Zhang@sanofi.com>
}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
data(esSim)
print(esSim)

# expression data
dat = exprs(esSim)
print(dim(dat))
print(dat[1:2,])

# phenotype data
pDat = pData(esSim)
print(dim(pDat))
print(pDat[1:2,])

# feature data
fDat = fData(esSim)
print(dim(fDat))
print(fDat[1:2,])

# choose the first 6 probes (3 OE probes, 2 UE probes, and 1 NE probe)
pDat$probe1 = dat[1,]
pDat$probe2 = dat[2,]
pDat$probe3 = dat[3,]
pDat$probe4 = dat[4,]
pDat$probe5 = dat[5,]
pDat$probe6 = dat[6,]

print(pDat[1:2, ])

pDat$grp = factor(pDat$grp)

# check histograms of probe 1 expression in cases and controls
print(table(pDat$grp, useNA = "ifany"))


statVisual(type = 'BiAxisErrBar', 
           dat = pDat, 
           group = 'grp', 
           y.left = 'probe1', 
           y.right = 'probe5') 

g1 = BiAxisErrBar( 
  dat = pDat, 
  group = 'grp', 
  y.left = 'probe1', 
  y.right = 'probe5') 


}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ method }% use one of  RShowDoc("KEYWORDS")

\name{BiAxisErrBar}
\alias{BiAxisErrBar}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Compare Groups Based on Scatter Plots
}
\description{
Compare groups based on two outcomes with different scales using error bar plots. Each bar plot displays mean \eqn{+/-}{+/-} standard error. 
}
\usage{
BiAxisErrBar(data, x, y1, y2, theme_classic = TRUE, title = NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
	  A data frame. Rows are subjects; Columns are variables describing the subjects.
}
  \item{x}{
	  character. A categorical variable in \code{data} that indicates the subject groups. The error bar plots will be drawn for each of the subject group. 
}
  \item{y1}{
	  character. The column name of \code{data} that indicates the base-scale variable for each subject group, the bar plot of which will be drawn on the left side.
}
  \item{y2}{
	  character. The column name of \code{data} that indicates the transformed-scale variable for each subject group, the bar plot of which will be drawn on the right side.
}
  \item{theme_classic}{
	  logical. Use classic background without grids (default: TRUE)
}
  \item{title}{
	  title of the plot
}
  \item{\dots}{
	  other input parameters for facet & theme.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
		A list with 9 elements.
	\code{data}, \code{layers}, \code{scales}, \code{mapping},
   	\code{theme}, \code{coordinates},
\code{facet}       \code{plot_env}, and \code{labels}.
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
           data = pDat, 
           x = 'grp', 
           y1 = 'probe1', 
           y2 = 'probe5', 
           title = 'Bi-axis error bar')

BiAxisErrBar( 
  data = pDat, 
  x = 'grp', 
  y1 = 'probe1', 
  y2 = 'probe5', 
  title = 'Bi-axis error bar')

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ method }% use one of  RShowDoc("KEYWORDS")

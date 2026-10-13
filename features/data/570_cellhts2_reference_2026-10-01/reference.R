# Data-only S4 adapter for unchanged upstream numerical functions. No fitting,
# residual, scaling or ranking algorithm is implemented in this adapter.
setClass('cellHTS', slots=c(values='array', geometry='numeric', plate_ids='integer',
                           annotations='character', state='list',
                           rowcol.effects='array', overall.effects='array'))
Data <- function(object) object@values
`Data<-` <- function(object,value) { object@values <- value; object }
pdim <- function(object) object@geometry
plate <- function(object) object@plate_ids
wellAnno <- function(object) object@annotations
state <- function(object) object@state
source('/perPlateScaling.R')
source('/adjustVariance.R')
x <- read.csv('/screen.csv', stringsAsFactors=FALSE)
stopifnot(nrow(x)==1536, all(table(x$plateID)==384))
output <- x[c('plateID','well','well_type')]
for (feature in c('cell_count_proxy','nuclear_area')) {
  object <- new('cellHTS', values=array(x[[feature]],dim=c(nrow(x),1L,1L)),
                geometry=c(nrow=16,ncol=24),
                plate_ids=as.integer(factor(x$plateID)),
                annotations=ifelse(x$well_type=='negcon','negative','sample'),
                state=list(configured=TRUE,normalized=FALSE))
  # These functions are sourced unmodified from pinned cellHTS2 source.
  fitted <- Bscore(object,save.model=TRUE)
  output[[paste0(feature,'_residual')]] <- as.vector(Data(fitted))
  output[[paste0(feature,'_plate')]] <- as.vector(adjustVariancebyPlate(fitted))
  output[[paste0(feature,'_pooled')]] <- as.vector(adjustVariancebyExperiment(fitted))
}
write.table(output,'/reference.csv',sep=',',row.names=FALSE,quote=TRUE,na='NA')
writeLines(c(R.version.string,capture.output(sessionInfo())),'/session.txt')

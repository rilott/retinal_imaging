
library(openssl)

read.csv('individuals.txt')->d

d$assign <- sample(c('train','test'),nrow(d),replace=T,prob=c(.8,.2))

for (i in 1:nrow(d)) {
individual <- d[i,'individual']
label <- d[i,'label']
assign <- d[i,'assign']
from <- Sys.glob(paste(file.path(label,individual),'*.png',sep=''))
to <- file.path(assign,label, unlist(lapply( strsplit( basename(from), '_'), function(x) { paste(paste(md5(x[1]),x[2],x[3],sep='_'),'.png',sep='') } )))
file.copy(from , to)
}

#write.csv(d,'individuals.txt',quote=FALSE,row.names=FALSE)


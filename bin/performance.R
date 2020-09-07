
library(data.table)

#run <- '2018-04-20-19-22-38'

for ( run in list.files(path='runs')) {
	print(run)
	for ( x in c('acc','loss')) {
		if (!file.exists(sprintf('grep %s runs/%s/valid.txt',x,run))) {
			next;
		}
		valid <- fread(sprintf('grep %s runs/%s/valid.txt',x,run))
		train <- fread(sprintf('grep %s runs/%s/train.txt',x,run))
		png(sprintf('%s-%s.png',x,run))
		plot(train$V2, train$V4, ylim=c(0,1), col='green',type='l')
		lines(valid$V2, valid$V4, col='red')
		dev.off()
	}
}



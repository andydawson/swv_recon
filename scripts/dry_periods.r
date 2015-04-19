recon = read.csv('csv/reconPrecip.csv')
colnames(recon) = c('year', 'recon')


plot(recon[,1], recon[,2], type='l')

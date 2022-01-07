p.mean1=c(5,5);
p.mean2=c(15,5);
p.mean3=c(10,10);
p.mean4=c(20,10);
p.mean5=c(5,15);
p.mean6=c(15,15);

p.sigma=matrix(c(2,0,0,2),2,2)
p.m=rbind(p.mean1,p.mean2,p.mean3,p.mean4, p.mean5, p.mean6)

n.mean1=c(10,5);n.mean2=c(20,5);n.mean3=c(5,10);n.mean4=c(15,10);n.mean5=c(10,15);n.mean6=c(20,15);
n.sigma=matrix(c(2.5,0,0,2.5),2,2)
n.m=rbind(n.mean1,n.mean2,n.mean3,n.mean4, n.mean5, n.mean6)

p.size.train=100;n.size.train=500; size.diff=n.size.train-p.size.train
p.size.test=10000;n.size.test=10000



p.index=sample(x=1:6, size=p.size.test, replace = T, prob = NULL)
n.index=sample(x=1:6, size=n.size.test, replace = T, prob = NULL)

p.test.x=matrix(0,p.size.test,2);n.test.x=matrix(0,n.size.test,2)
for(i in 1:p.size.test) p.test.x[i,]=rmvnorm(1,mean=p.m[p.index[i],],sigma=p.sigma)
for(i in 1:n.size.test) n.test.x[i,]=rmvnorm(1,mean=n.m[n.index[i],],sigma=n.sigma)

test.x=rbind(p.test.x,n.test.x);p.test.y=rep("pos",p.size.test); n.test.y=rep("neg",n.size.test)
test.y=c(p.test.y, n.test.y)
test.data=data.frame(test.x,test.y);colnames(test.data)=c("x1","x2","y")
print(test.data)

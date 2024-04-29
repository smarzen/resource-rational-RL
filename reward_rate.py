import numpy as np
import scipy as sp
import numpy.linalg as LA
import matplotlib.pyplot as plt

# create environment
aS = 1; aW = 1
M = 50
N = 40
W = 20
# draw the reward function
r = np.random.uniform(size=[W,M])
# draw the causal state distribution
psigma = np.random.dirichlet(alpha=aS*np.ones(N),size=1).T
# draw the distribution of w given belief states
pwgsigma = np.random.dirichlet(alpha=aW*np.ones(W),size=N)

def getFeatures(r,psigma,pwgsigma,beta,lambd):
	W, M = r.shape
	N = len(psigma)
	# initialize arrays
	k = N
	psgsigma = np.random.dirichlet(alpha=np.ones(k),size=N)
	piags = np.random.dirichlet(alpha=np.ones(M),size=k)
	# run GBA algorithm
	for i in range(20000): # fix this
		# update piags
		ps = np.dot(psgsigma.T,psigma) # check this
		pia = np.dot(piags.T,ps)
		psigmags = np.dot(np.diag(1/ps[:,0]),np.dot(psgsigma.T,np.diag(psigma[:,0])))
		fooa = np.log(np.meshgrid(pia,np.ones(k))[0]) + (1/beta)*np.dot(np.dot(psigmags,pwgsigma),r) # check this
		Za = np.sum(np.exp(fooa),1)
		fooa = fooa - np.log(np.meshgrid(Za,np.ones(M))[0]).T
		# update psgsigma
		foos = np.log(np.meshgrid(ps,np.ones(N))[0]).T+(1/lambd)*np.dot(piags,np.dot(pwgsigma,r).T) # check this
		Zs = np.sum(np.exp(foos),0)
		foos = foos - np.log(np.meshgrid(Zs,np.ones(k))[0])
		#
		piags = np.exp(fooa)
		psgsigma = np.exp(foos).T
	# calculate reward and rates
	reward = np.sum(np.dot(piags.T,np.dot(np.dot(np.diag(psigma[:,0]),psgsigma).T,pwgsigma))*r.T)
	ps = np.dot(psgsigma.T,psigma)
	pia = np.dot(piags.T,ps)
	ratea = -np.nansum(pia*np.log2(pia))+np.nansum(np.dot(ps[:,0],np.nansum(piags*np.log2(piags),1)))
	rates = -np.nansum(ps*np.log2(ps))+np.nansum(np.dot(psigma[:,0],np.nansum(psgsigma*np.log2(psgsigma),1)))
	return reward, ratea, rates

# visualize the surface
# sweep over all beta and lambd
rewards = []
rates_a = []
rates_s = []
for beta in np.linspace(0.01,1,1000):
	for lambd in np.linspace(0.01,1,1000):
		a, b, c = getFeatures(r,psigma,pwgsigma,beta,lambd)
		rewards.append(a)
		rates_a.append(b)
		rates_s.append(c)
		np.savez('Reward-rate_function_20000iterations.npz',aS=aS,aW=aW,M=M,N=N,W=W,r=r,psigma=psigma,pwgsigma=pwgsigma,
			rewards=rewards,rates_a=rates_a,rates_s=rates_s)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(rates_s, rates_a, rewards, marker='o')
ax.set_xlabel("Sensory rate")
ax.set_ylabel("Actuator rate")
ax.set_zlabel("Reward")
plt.show()




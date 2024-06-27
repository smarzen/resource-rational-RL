import numpy as np
import scipy as sp
import numpy.linalg as LA
import matplotlib.pyplot as plt

# create environment
aO = 1; aW = 1
M = 5
N = 4
W = 3
O = 2
# draw the reward function
r = np.random.uniform(size=[W,M])
# draw the observation probabilities
pogw = np.random.dirichlet(alpha=aO*np.ones(O),size=W)
# draw the transition probabilities from a Dirichlet distribution
pwgaw = np.random.dirichlet(alpha=aW*np.ones(W),size=(M,W))

def getFeatures(r,pogw,pwgaw,N,beta,lambd,L,eps):
	W, M = r.shape
	foo, O = pogw.shape
	# initialize arrays
	k = 3
	psgsigma_used = np.random.dirichlet(alpha=np.ones(k),size=N) # arbitrary initial # of causal states
	# initialize psigma and pwgsigma
	psigma_used = np.random.dirichlet(alpha=np.ones(N))
	pwgsigma_used = np.random.dirichlet(alpha=np.ones(W),size=N)
	psgsigma = {}
	psigma = {}
	pwgsigma = {}
	for i in range(N):
		ind = np.asarray([i,i,i]).astype(int)
		psgsigma[str(ind)] = psgsigma_used[i,:] # just to start us off
		psigma[str(ind)] = psigma_used[i]
		pwgsigma[str(ind)] = pwgsigma_used[i,:]
	piags = np.random.dirichlet(alpha=np.ones(M),size=k)
	def getcausalstates(psgsigma,psgsigma_used,piags,r,pogw,pwgaw,psigma,psigma_used,pwgsigma,pwgsigma_used,L,eps):
		pagsigma = np.dot(psgsigma_used,piags) # check this
		pa = np.dot(psigma_used,pagsigma)
		# first get T^{(o)}
		Ts = {}
		for o in range(O):
			paogw = np.outer(pogw[:,o],pa)
			final_matrix = np.zeros([W,W])
			for w in range(W):
				for w2 in range(W):
					final_matrix[w,w2] = np.dot(pwgaw[:,w2,w],paogw[w2,:])
			Ts[str(o)] = final_matrix
		# then get T and eig_1(T) to get mu
		T = np.zeros([W,W])
		for o in range(O):
			T = T+Ts[str(o)]
		w, v = np.linalg.eig(T.T)
		mask = np.abs(w-1)<1e-5
		foo = v[:,mask]
		mu = np.real(foo)/np.sum(np.real(foo))
		mu = mu
		# then get p(w|sigma) from Paul's iteration
		# run over all observation sequences of length L
		obs_sequences = {}
		for num in range(np.power(O,L)):
			# express in base O
			foo = np.base_repr(num,base=O)
			# pad it
			if len(foo)<L:
				pad = '0'*(L-len(foo))
			foo = pad+foo
			foo = [int(x) for x in foo]
			# iteratively apply T's
			start_foo = mu
			for j in foo:
				start_foo = np.dot(Ts[str(j)],mu)
				start_foo = start_foo/np.sum(start_foo)
			obs_sequences[str(num)] = start_foo
		# then get p(sigma) from coarse-graining with nmpf
		# basically now all the obs_sequences correspond to a belief state
		# go to
		pwgsigma = {}
		# first, fill in Grid0
		psigma = {}
		for key in obs_sequences:
			state = obs_sequences[key]
			# find index of state as floor(state/eps)
			ind = np.floor(state/eps)
			ind = ind.astype(int).T[0,:]
			# find all values that go here
			if str(ind) in pwgsigma:
				n = psigma[str(ind)]+1
				pwgsigma[str(ind)] = ((n-1)/n)*pwgsigma[str(ind)]+(1/n)*state
				psigma[str(ind)] += 1
			else:
				pwgsigma[str(ind)] = state
				psigma[str(ind)] = 1
		# normalize psigma
		Z = np.sum(np.asarray(list(psigma.values())))
		for key in psigma:
			psigma[key] = psigma[key]/Z
		# find new psgsigma
		# find all the keys and then associate for each key the right sigma
		psigma_used = []
		pwgsigma_used = []
		psgsigma_used = []
		psgsigma_new = {}
		for key in psigma:
			# that key is the sensorimotor causal state
			psigma_used.append(psigma[key])
			pwgsigma_used.append(pwgsigma[key])
			# find the closest key in psgsigma and assign
			dist = 100000
			keyopt = 0
			for key2 in psgsigma:
				belief1 = np.asarray([int(x) for x in str(key)[1:-1].split()])
				belief2 = np.asarray([int(x) for x in str(key2)[1:-1].split()])
				dist_new = np.sum((belief1-belief2)**2)
				if dist_new<dist:
					keyopt = key2
			psgsigma_new[key] = psgsigma[keyopt]
			psgsigma_used.append(psgsigma[keyopt])
		return psigma, pwgsigma, psgsigma_new, np.asarray(psigma_used), np.asarray(pwgsigma_used)[:,:,0], np.asarray(psgsigma_used)
	# run GBA algorithm
	for i in range(200): # fix this
		N = len(psigma_used)
		# update piags
		ps = np.dot(psgsigma_used.T,psigma_used) # check this
		pia = np.dot(piags.T,ps)
		psigmags = np.dot(np.diag(1/ps),np.dot(psgsigma_used.T,np.diag(psigma_used)))
		fooa = np.log(np.meshgrid(pia,np.ones(k))[0]) + (1/beta)*np.dot(np.dot(psigmags,pwgsigma_used),r) # check this
		Za = np.sum(np.exp(fooa),1)
		fooa = fooa - np.log(np.meshgrid(Za,np.ones(M))[0]).T
		# update psgsigma
		foos = np.log(np.meshgrid(ps,np.ones(N))[0]).T+(1/lambd)*np.dot(piags,np.dot(pwgsigma_used,r).T) # check this
		Zs = np.sum(np.exp(foos),0)
		foos = foos - np.log(np.meshgrid(Zs,np.ones(k))[0])
		#
		piags = np.exp(fooa)
		psgsigma_used = np.exp(foos).T
		# get the next iteration of psigma and pwgsigma
		psigma, pwgsigma, psgsigma, psigma_used, pwgsigma_used, psgsigma_used = getcausalstates(psgsigma,psgsigma_used,piags,r,pogw,pwgaw,psigma,
			psigma_used,pwgsigma,pwgsigma_used,L,eps)
	# calculate reward and rates
	reward = np.sum(np.dot(piags.T,np.dot(np.dot(np.diag(psigma_used),psgsigma_used).T,pwgsigma_used))*r.T)
	ps = np.dot(psgsigma_used.T,psigma_used)
	pia = np.dot(piags.T,ps)
	ratea = -np.nansum(pia*np.log2(pia))+np.nansum(np.dot(ps,np.nansum(piags*np.log2(piags),1)))
	rates = -np.nansum(ps*np.log2(ps))+np.nansum(np.dot(psigma_used,np.nansum(psgsigma_used*np.log2(psgsigma_used),1)))
	return reward, ratea, rates

def maximiner(r,pogw,pwgaw):
	W, M = r.shape
	foo, O = pogw.shape
	# find g
	reward = 0
	ratea = 0
	rates = 0
	for i in range(np.power(M,W)):
		foo = np.base_repr(i,base=M)
		# pad it
		if len(foo)<W:
			pad = '0'*(W-len(foo))
		foo = pad+foo
		foo = [int(x) for x in foo]
		# calculate the p_g(w)
		T = np.zeros([W,W])
		for j in range(W):
			T[:,j] = pwgaw[int(foo[j]),j,:]
		w, v = np.linalg.eig(T)
		mask = np.abs(w-1)<1e-5
		pg = v[:,mask]
		pg = np.real(pg)
		pg = pg/np.sum(pg)
		rates_new = -np.nansum(pg*np.log2(pg))
		# coarse grain to get the ratea
		pa = {}
		for j in range(W):
			if str(foo[j]) in pa:
				pa[str(foo[j])] += pg[j]
			else:
				pa[str(foo[j])] = pg[j]
		pa = np.asarray(list(pa.values()))
		ratea_new = -np.nansum(pa*np.log2(pa))
		# get the reward
		rs = np.zeros(W)
		for j in range(W):
			rs[j] = r[j,foo[j]]
		reward_new = np.sum(pg*rs)
		# then replace if better
		if reward_new>reward:
			reward = reward_new
			ratea = ratea_new
			rates = rates_new
	return reward, ratea, rates

# visualize the surface
# sweep over all beta and lambd
# rewards = []
# rates_a = []
# rates_s = []
# for beta in np.linspace(0.01,1,1000):
# 	for lambd in np.linspace(0.01,1,1000):
# 		a, b, c = getFeatures(r,psigma,pwgsigma,beta,lambd)
# 		rewards.append(a)
# 		rates_a.append(b)
# 		rates_s.append(c)
# 		np.savez('Reward-rate_function_20000iterations.npz',aS=aS,aW=aW,M=M,N=N,W=W,r=r,psigma=psigma,pwgsigma=pwgsigma,
# 			rewards=rewards,rates_a=rates_a,rates_s=rates_s)

# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(rates_s, rates_a, rewards, marker='o')
# ax.set_xlabel("Sensory rate")
# ax.set_ylabel("Actuator rate")
# ax.set_zlabel("Reward")
# plt.show()

r1, Ra, Rs = getFeatures(r,pogw,pwgaw,1,0.01,0.01,10,0.01)
print(r1, Ra, Rs)
rmm, Ramm, Rsmm = maximiner(r,pogw,pwgaw)
print(rmm, Ramm, Rsmm)


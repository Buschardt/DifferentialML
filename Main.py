#%%
import Models
import NN

import numpy as np
#%%
#test
print(Models.bsDelta(100, 110, 0.05, 1))

#%%
##BS European Call
# simulation set sizes to perform
sizes = [1024, 8192]

# show delta?
showDeltas = True

# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000) 
print("using seed %d" % simulSeed)
weightSeed = None

# number of test scenarios
nTest = 100    

# go
generator = Models.BlackScholes()
xAxis, yTest, dydxTest, vegas, values, deltas = \
    NN.test(generator, sizes, nTest, simulSeed, None, weightSeed)

#%%
# show predicitions
NN.graph("Black & Scholes", values, xAxis, "", "values", yTest, sizes, True)

# show deltas
if showDeltas:
    NN.graph("Black & Scholes", deltas, xAxis, "", "deltas", dydxTest, sizes, True)

#%%
##Bachelier Basket dim 1
# basket / bachelier dimension
basketDim = 1

# simulation set sizes to perform
sizes = [1024, 8192]

# show delta?
showDeltas = True
deltidx = 0 # show delta to first stock

# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000) 
print("using seed %d" % simulSeed)
testSeed = None
weightSeed = None
    
# number of test scenarios
nTest = 4096    

# go
generator = Models.Bachelier(basketDim)
xAxis, yTest, dydxTest, vegas, values, deltas = \
    NN.test(generator, sizes, nTest, simulSeed, None, weightSeed)

#%%
# show predicitions
NN.graph("Bachelier dimension %d" % basketDim, values, xAxis, "", "values", yTest, sizes, True)

# show deltas
if showDeltas:
    NN.graph("Bachelier dimension %d" % basketDim, deltas, xAxis, "", "deltas", dydxTest, sizes, True)

#%%
##Bachelier Basket dim 7
# basket / bachelier dimension
basketDim = 7

# simulation set sizes to perform
sizes = [4096, 8192, 16384]

# show delta?
showDeltas = True
deltidx = 0 # show delta to first stock

# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000) 
print("using seed %d" % simulSeed)
testSeed = None
weightSeed = None
    
# number of test scenarios
nTest = 4096    

# go
generator = Models.Bachelier(basketDim)
xAxis, yTest, dydxTest, vegas, values, deltas = \
    NN.test(generator, sizes, nTest, simulSeed, None, weightSeed)

#%%
# show predicitions
NN.graph("Bachelier dimension %d" % basketDim, values, xAxis, "", "values", yTest, sizes, True)

# show deltas
if showDeltas:
    NN.graph("Bachelier dimension %d" % basketDim, deltas, xAxis, "", "deltas", dydxTest, sizes, True)

#%%
##Bachelier Basket dim 20
# basket / bachelier dimension
basketDim = 20

# simulation set sizes to perform
sizes = [4096, 8192, 16384]

# show delta?
showDeltas = True
deltidx = 0 # show delta to first stock

# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000) 
print("using seed %d" % simulSeed)
testSeed = None
weightSeed = None
    
# number of test scenarios
nTest = 4096    

# go
generator = Models.Bachelier(basketDim)
xAxis, yTest, dydxTest, vegas, values, deltas = \
    NN.test(generator, sizes, nTest, simulSeed, None, weightSeed)

#%%
# show predicitions
NN.graph("Bachelier dimension %d" % basketDim, values, xAxis, "", "values", yTest, sizes, True)

# show deltas
if showDeltas:
    NN.graph("Bachelier dimension %d" % basketDim, deltas, xAxis, "", "deltas", dydxTest, sizes, True)
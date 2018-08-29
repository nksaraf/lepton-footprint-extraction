import model.connections as con

a = con.LambdaTransformer((lambda x, y: str(x) + '__' + str(y)), 'under')  
b = con.LambdaTransformer((lambda x, y: str(x) + '///' + str(y)), 'slash')
z = a + b

d = con.LambdaTransformer((lambda x, y: str(x) + '***' + str(y)), 'mul')

w = con.Wire(x=10, y=8)

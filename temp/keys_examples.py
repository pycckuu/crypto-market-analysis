'''
### The example of the code to use the public and private RSA ans SHA keys
'''

'''
Importing required libraries
'''
#%%
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP

'''
Generate secrete message and keys
'''
#%%
secret_message = b'Can you see this password?'
key = RSA.generate(1024)
pub_key = key.publickey()
private_key = key.exportKey()

'''
Encrypt the message
'''
#%%
cipher = PKCS1_OAEP.new(pub_key)
ciphertext = cipher.encrypt(secret_message)

'''
Decrypt the message
'''
#%%
decipher =PKCS1_OAEP.new(key)
print(decipher.decrypt(ciphertext))

''' Signing the messages '''
#%%
from Cryptodome.Hash import SHA256

''' Generate Hash code'''
#%%
h = SHA256.new(secret_message)
hd =h.hexdigest()
print(hd)

''' sign the message '''
#%%
from Cryptodome.Signature import PKCS1_v1_5

signer = PKCS1_v1_5.new(key)
signature = signer.sign(h)
print(signature)

''' verify the signature '''
#%%
PKCS1_v1_5.new(pub_key).verify(h, signature)
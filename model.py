""" We want to build a transformer model that can translate from some input integral of the form:

input = {a1,a2,...,an}

and maps to some output:

output = {I1,c1,I2,c2,...}

where ai and ci are base-10 integers, and I1 individually tokenized integrals. 
This will require a number of steps. 

TODO list:
1. tokenizer/detokenizer (take string to token sequence)
2. padding function (so all inputs and outputs have same length)
3. transformer model (with encoders and decoders)
4. training function
5. testing function
6. Dataloader for batching the data

"""

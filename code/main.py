from building import *
from testing import *
from training import *

tokenizer = CustomTokenizer()

# --------------
# Hyperparamters
# --------------
vocab_size = len(tokenizer.vocab)
d_model = 512
d_ffn = 512
nhead = 8
encoder_layers = 4
decoder_layers = 4
dropout = 0.1
PAD_TOKEN_ID = 0


# --------------
# MAIN
# --------------
if __name__ == "__main__":

    # ---------------------
    # Model & Data
    # ---------------------

    model = KernelModel(vocab_size,d_model,d_ffn,nhead,encoder_layers,decoder_layers,dropout)
    data = FIRE6Dataset("../data/train", tokenizer, max_len=850)
    batchsize = 5
    dataloader = DataLoader(data,batch_size=batchsize,shuffle=True)

    # ---------------------
    # Trainng Objects & Device
    # ---------------------
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    model.to(device)  

    # ---------------------
    # Training Routine
    # ---------------------
    
    train(model, device, criterion, 0.0001, dataloader, epochs, "TC_KernelModel_D512_H8_L4")
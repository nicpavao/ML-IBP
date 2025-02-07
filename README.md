# Dev Box for IBP Transormer
I (Nic) will be pushing code that works to this directory. 
## SETUP
First, setup the virtual environment and run the requirements file:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Update pip if outdated.
## OVERVIEW
I'm making a tutorial jupyter notebook that demonstrates what the model is and how it works, 
i.e. what is the backend, and how does it perform string translation.

It's also split into 4 subdirectories:

1. code
2. data
3. models
4. analysis

CODE: contains a number of python files needed for
1. building
    - boilerplate for KernelModel (what we train)
        - EncoderStack
        - DecoderStack
    - boilerplate for ProjectionModel (what we inference)
    - dataloading and tokenizers are here
2. training
    - training run that uses teacher forcing to train KernelModel 
3. testing
    - testing run on tokens (queries the KernelModel)
    - testing run for sequences (queries the ProjectionModel)
4. running model predictions

DATA: a variety of processed and prepocessed data
- training data (pre-processed, strings, .dat files)
- training data (processed, integer lists, .json files)
- testing data (pre-processed, strings, .dat files)
- testing data (processed, integer lists, .json files)

MODELS: a directory for models that have been trained
- model path files
- README.md that summarizes what each models does, hypers, performance, etc.

ANALYSIS: there are a bunch of data visuals and plots we should assemble.
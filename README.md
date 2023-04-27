# Sementic-Segmentation-With-Diffusion-Models
Forked project from Julia Wolleb project : https://github.com/JuliaWolleb/Diffusion-based-Segmentation

This is a improved version that allow Semantic Segmentation and more general framework.

## Usage 
You can train a diffusion model using the file train.sh
Then you can sample using sample.sh

Please modify those files to modify the training/smapling behavior (learning 
rate, number of samples ...) 

During sampling, sampled will be shown using the Vizdom package on the port 6006. 
To avoid bad behavior, it is recommended to start a Vizdom Server on a other terminal 
using the command : 

```
python3 -m visdom.server -port 6006
```

TO DO : 

Adpat to another dataset (on going) 
make sure their no miss with dimensions (by mb cange number of input and ouput channels) 
make a quick report with vizualisation 

chnage vizdom shit for tenosrboard image



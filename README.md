# Sementic-Segmentation-With-Diffusion-Models
Forked project from Julia Wolleb project : https://github.com/JuliaWolleb/Diffusion-based-Segmentation

This is a improved version that allow Semantic Segmentation and more general framework.

## Usage 
Training a diffusion model for Semantic Segmentation can be done by calling,
__segmentation_train.py__ file. 

To do it on your data, you need to define a Dataset object (see torch.data.Dataset or the section below)
and edit __segmentation_train.py__.

A convenient way of making calls to __segmentation_train.py__ is to use a .sh script. 
Example is show in __train.sh__.

Then, once you have a trained model, (that should be place in *./result/\*_nb_training_step.pt* by default),
you can use __segmentation_sample.py__ to generate segmentation masks with this model.

Again, you will need to edit that file to define your own Dataset.

Example of calling __segmentation_sample.py__ is show in __sample.sh__

During sampling, you can visualize the denoising process with tensorboard by passing the __--tb_display__
parameter to __segmentation_sample.py__. 

Sampled exemples will be saved as well in files (only finals samples) in *./sampling_[datetime]*.
You can use __--save_numpy__ to save raw model outputs (multi channels image) in *./sampling_[datetime]/numpy_arrays*. 
To have a better visualization of the segmentation masks, you can pass __--save_nice__ to save 
a nice verion of masks with colors for each classes *./sampling_[datetime]/niced*.

## Defining your own Dataset 
```python

class MyDataset(torch.utils.data.Dataset) : 

  def __init__ (self, ..., test_flag)
    self.test_flag = test_flag
    ...
  def __len__(self) : 
    ...
  def __getitem__(self, idx) : 
    ...
    if self.test_flag : 
      return image, image_indice_or_name
    else : 
      return image, label 
```

## TO DO 
Pillow import si redondent w/ scikitimage, ditch one of them
Add data, results, nince vizaulisations

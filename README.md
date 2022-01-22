# JAX Perlmutter Tutorials
Collection of tips and tutorials for running [JAX](https://github.com/google/jax) on [Perlmutter](https://www.nersc.gov/systems/perlmutter/)

## Installing JAX on Perlmutter (Jan. 2022)

#### Installing JAX in the default python environment

Installing JAX on Perlmutter is easy if you follow these steps:
```bash
module load python cudnn/8.2.0 nccl/2.11.4 cudatoolkit/21.9_11.4 
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
And that's it, but note that to run properly, JAX will require that you load the following modules: `cudnn/8.2.0` `nccl/2.11.4` `cudatoolkit/21.9_11.4`. 

#### Making JAX available in JupyterLab

To make sure the necessary modules are loaded when you run your notebooks on JupyterLab, you will then need to create a custom Jupyter kernel.

1. **Create a template kernel**
```bash
python -m ipykernel install --user --name jax --display-name JAX
```
This will create a template kernel named `JAX`, which we now need to modify slightly.

Go to the newly created kernel directory:
```
cd $HOME/.local/share/jupyter/kernels/jax
ls
```
You should see in this directory a `kernel.json` which we will edit in the next step.

2. **Edit kernel with custom startup script**

Open the `kernel.json` file and edit to the following:
```json
{
 "argv": [
  "{resource_dir}/kernel-helper.sh",
  "python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "JAX",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```
Now, in addition, create a new file in the same directory named `kernel-helper.sh` with the following content:
```bash
#!/bin/bash -l
module load python cudnn/8.2.0 nccl/2.11.4 cudatoolkit/21.9_11.4
exec "$@"
```

And that should be it. Now when you launch the JAX kernel on Perlmutter you should be able to run your jax code without issue.

# Example of the C API for tensorflow

This is very much based on the example from: https://github.com/AmirulOm/tensorflow_capi_sample
It will describe things :)

## You can run the example with:

I wanted to simplify the example even further and add more memory cleanups. So you can build and run this example with:

````
make run
````

The dependencies should fairly much be:

````
gcc make tensorflow
````

If you want to see the commands to run this example, run:

````
make run -B -n
````

The code will download the library neccessary and compile the code for you.
You will need the tensorflow library installed on your computer.
If you are with windows, i would strongly recommend to build this with msys2. Otherwise build this in a docker environment. 
I built this with Archlinux.

## Extract model information:

using `serving_default` signature key into command to print out the tensor node:
```
saved_model_cli show --dir <path_to_saved_model_folder> --tag_set serve --signature_def serving_default
```

# Examples

## demo1_very_simple

Will demonstrate how to use the C API in basic terms. Like this input and output tensors.

## demo2_simple_train

Will show the keras addition example which will train and generate a model which we will use with the C API

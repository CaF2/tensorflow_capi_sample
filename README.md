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

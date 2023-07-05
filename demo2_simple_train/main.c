/**
Based on https://github.com/AmirulOm/tensorflow_capi_sample

Copyright (c) 2023 Florian Evaldsson <florian.evaldsson@telia.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "tensorflow/c/c_api.h"

const char *TRANSLATION_TBL=" +0123456789";
size_t TRANSLATION_TBL_LEN=12;

void no_op_deallocator(void *data, size_t a, void *b)
{
	
}

/**
	Convert a string to something we can use for the input tensor.
	
	 @param c
	 	Input character
	 @param data
	 	contents of a tensor cell, in this case float[12]
	 @returns
	 	0=success
*/
int set_value_from_char(char c, float *data)
{
	int isset=0;
	
	for(int i=0;i<TRANSLATION_TBL_LEN;i++)
	{
		if(TRANSLATION_TBL[i]==c)
		{
			data[i]=1.0;
			isset=1;
		}
		else
		{
			data[i]=0.0;
		}
	}
	
	if(!isset)
	{
		//set unknowns to space
		data[0]=1.0;
	}
	
	return !isset;
}

/**
	Convert a string to something we can use for the input tensor.
	
	 @param input
	 	a string of any size, will add spaces to the end, which is really not what the example is trained for.
	 @param data
	 	contents of the tensor. In this case the size float[1*7*12]
	 @returns
	 	0=success
*/
int set_value_from_string(const char *input, float *data)
{
	for(const char *tmp=input;*tmp;tmp++)
	{
		int i=tmp-input;
		if(i>=7)
		{
			return 1;
		}
		set_value_from_char(*tmp,data+(12*i));
	}
	
	return 0;
}

/**
	We will convert a result tensor to a output string value. We may get:
	
	0.000382 0.000001 0.000303 0.000079 0.001206 0.000247 0.009572 0.505685 0.440370 0.039955 0.002178 0.000023 
	0.999988 0.000000 0.000000 0.000000 0.000006 0.000005 0.000001 0.000000 0.000000 0.000000 0.000000 0.000000 
	1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
	1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
	
	if one value is acceptable, we will convert it to 1 and the rest to 0
	
	 @param data
	 	input data of the size float[1*4*12]
	 @param output
	 	string of size 4+1
	 @returns
	 	0=success
*/
int get_string_from_value(float *data,char *output)
{
	for(size_t str=0;str<4;str++)
	{
		size_t highest=-1;
		
		for(size_t pos=0;pos<12;pos++)
		{
			//we say this is an acceptable score
			if(data[str*12 + pos]>0.3 && (highest==-1 || data[str*12 + pos] > data[str*12 + highest]))
			{
				highest=pos;
/*				output[str]=TRANSLATION_TBL[pos];*/
/*				break;*/
			}
		}
		
		if(highest>=0)
		{
			output[str]=TRANSLATION_TBL[highest];
		}
		else
		{
			output[str]=' ';
		}
	}
	
	return 0;
}

int main(const int argc, char **const argv)
{
	//********* Read model
	TF_Graph *graph = TF_NewGraph();
	TF_Status *status = TF_NewStatus();

	TF_SessionOptions *session_opts = TF_NewSessionOptions();
	TF_Buffer *run_opts = NULL;

	const char *saved_model_dir = "add_rnn/";
	const char *tags = "serve"; // default model serving tag; can change in future
	int ntags = 1;

	TF_Session *session = TF_LoadSessionFromSavedModel(session_opts, run_opts, saved_model_dir, &tags, ntags, graph, NULL, status);
	if (TF_GetCode(status) == TF_OK)
	{
		fprintf(stdout,"%s:%d TF_LoadSessionFromSavedModel [OK]\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stderr,"%s:%d TF_LoadSessionFromSavedModel [%s]\n",__FILE__,__LINE__,TF_Message(status));
	}

	//****** Get input tensor
	// TODO : need to use saved_model_cli to read saved_model arch
	int num_inputs = 1;
	TF_Output input[num_inputs];
	
	//Check also TF_GraphNextOperation()
	TF_Output t0 = {TF_GraphOperationByName(graph, "serving_default_lstm_input"), 0};
	if (t0.oper == NULL)
	{
		fprintf(stderr,"%s:%d ERROR: Failed TF_GraphOperationByName serving_default_lstm_input\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stdout,"%s:%d TF_GraphOperationByName serving_default_lstm_input is OK\n",__FILE__,__LINE__);
	}

	input[0] = t0;

	//********* Get Output tensor
	int num_outputs = 1;
	TF_Output output[num_outputs];
	
	//Check also TF_GraphNextOperation()
	TF_Output t2 = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
	if (t2.oper == NULL)
	{
		fprintf(stderr,"%s:%d ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stdout,"%s:%d TF_GraphOperationByName StatefulPartitionedCall is OK\n",__FILE__,__LINE__);
	}

	output[0] = t2;

	//********* Allocate data for inputs & outputs
	TF_Tensor *input_values[num_inputs];
	TF_Tensor *output_values[num_outputs];

	int ndims = 3;
	int64_t dims[] = {1, 7, 12};
	
	/*
	Example of a input table. We will convert it to a string
	[
		[ True False False False False False False False False False False False]
		[False False False False False False False False False  True False False]
		[False False False False False False False False  True False False False]
		[False  True False False False False False False False False False False]
		[False False False False False False False False False False  True False]
		[False False False False False False False False False False False  True]
		[False False False False False False False False False False  True False]
	]
	-> [ 76+898]
	*/
	float data[1 * 7 * 12];
/*	for (int i = 0; i < (1 * 7 * 12); i++)
	{
		data[i] = 1.00;
	}*/
	
	const char *input_data=" 273+92";
	//const char *input_data=" 273+92";
	//const char *input_data="    2+2";
	set_value_from_string(input_data,data);
	int ndata = sizeof(float) * 7 * 12; // This is tricky, it number of bytes not number of element

	TF_Tensor *int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &no_op_deallocator, 0);
	if (int_tensor != NULL)
	{
		fprintf(stdout,"%s:%d TF_NewTensor is OK\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stderr,"%s:%d ERROR: Failed TF_NewTensor\n",__FILE__,__LINE__);
	}

	input_values[0] = int_tensor;

	// //Run the Session
	TF_SessionRun(session, NULL, input, input_values, num_inputs, output, output_values, num_outputs, NULL, 0, NULL, status);

	if (TF_GetCode(status) == TF_OK)
	{
		fprintf(stdout,"%s:%d Session is OK\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stderr,"%s:%d Session returned with [%s]\n",__FILE__,__LINE__, TF_Message(status));
	}
	
	// //Free memory
	TF_DeleteGraph(graph);
	TF_DeleteSession(session, status);
	TF_DeleteSessionOptions(session_opts);
	TF_DeleteStatus(status);

	void *buff = TF_TensorData(output_values[0]);
	int64_t num_elements = TF_TensorElementCount(output_values[0]);
	
	float *offsets = (float *)buff;
	printf("Result Tensor %lld:\n",num_elements);
	for (int i = 0; i < num_elements; i++)
	{
		if(i%12==0)
		{
			printf("\n");
		}
	
		printf("%f ", offsets[i]);
	}
	
	printf("\n");
	
	//need '\0' in the end
	char output_s[4+1]={0};
	
	get_string_from_value(offsets,output_s);
	
	fprintf(stdout,"%s:%d OUTPUT VALUE [%s] = [%s]\n",__FILE__,__LINE__,input_data,output_s);
	
	TF_DeleteTensor(int_tensor);
	TF_DeleteTensor(output_values[0]);
}

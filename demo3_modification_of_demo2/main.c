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

const char *TRANSLATION_TBL=" +-0123456789";
size_t TRANSLATION_TBL_LEN=13;

//see invert from the python code.
int INVERT = 1;

void no_op_deallocator(void *data, size_t a, void *b)
{
	
}

/**
	Convert a string to something we can use for the input tensor.
	
	@param c
		Input character
	@param data
		contents of a tensor cell, in this case float[TRANSLATION_TBL_LEN]
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
		contents of the tensor. In this case the size float[1*7*TRANSLATION_TBL_LEN]
	@returns
		0=success
*/
int set_value_from_string(const char *input, float *data)
{
	int i=7;

	for(const char *tmp=input;*tmp;tmp++)
	{
		i=tmp-input;
		if(i>=7)
		{
			return 1;
		}
		
		const int curr_i=INVERT?(7-i-1):i;
		set_value_from_char(*tmp,data+(TRANSLATION_TBL_LEN*curr_i));
	}
	
	i++;
	
	while(i<7)
	{
		const int curr_i=INVERT?(7-i-1):i;
		set_value_from_char(' ',data+(TRANSLATION_TBL_LEN*curr_i));
	
		i++;
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
		input data of the size float[1*4*TRANSLATION_TBL_LEN]
	@param output
		string of size 4+1
	@returns
		0=success
*/
int get_string_from_value(float *data,char *output, size_t max_num, int inversed)
{
	for(size_t str=0;str<max_num;str++)
	{
		const size_t curr_i=inversed?(max_num-str-1):str;
	
		size_t highest=-1;
		
		for(size_t pos=0;pos<TRANSLATION_TBL_LEN;pos++)
		{
			//we say this is an acceptable score
			if(data[str*TRANSLATION_TBL_LEN + pos]>0.3 && (highest==-1 || data[str*TRANSLATION_TBL_LEN + pos] > data[str*TRANSLATION_TBL_LEN + highest]))
			{
				highest=pos;
			}
		}
		
		if(highest>=0)
		{
			output[curr_i]=TRANSLATION_TBL[highest];
		}
		else
		{
			output[curr_i]=' ';
		}
	}
	
	return 0;
}

/**
	Convert a string of the form "A+B" to the result
	
	@param input
		input string
	@returns
		value, if failed -1.
*/
long get_actual_result(const char *input)
{
	for(char *tmp=(char*)input;*tmp;tmp++)
	{
		if(*tmp>='0' && *tmp<='9')
		{
			long a_val=strtol(tmp,&tmp,10);
			
			if(*tmp=='+')
			{
				long b_val=strtol(tmp+1,NULL,10);
				
				return a_val+b_val;
			}
			else if(*tmp=='-')
			{
				long b_val=strtol(tmp+1,NULL,10);
				
				return a_val-b_val;
			}
		}
	}
	
	return -1;
}

/**
	Print content of tensor
	
	@param data
		tensor data
	@param num_elements
		number of elements
*/
int print_tensor(float *data, size_t num_elements)
{
	for (int i = 0; i < num_elements; i++)
	{
		if(i%TRANSLATION_TBL_LEN==0 && i!=0)
		{
			printf("\n");
		}
	
		printf("%f ", data[i]);
	}
	
	printf("\n");
	
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
	int64_t dims[] = {1, 7, TRANSLATION_TBL_LEN};
	float data[1 * 7 * TRANSLATION_TBL_LEN];
	
	char input_data[7+1]={0};
	int input_data_pos=0;
	
	int exit=0;
	
	while(exit==0)
	{
		printf("Input simple equation (q to quit):\n");	
		
		while(1)
		{
			char c=getc(stdin);
			
			if(c=='\n')
			{
				if(input_data_pos>0)
				{
					break;
				}
				else
				{
					input_data_pos=0;
					continue;
				}
			}
			else if(c=='q')
			{
				exit=1;
				goto quit_prog;
			}
			input_data[input_data_pos]=c;
			input_data_pos++;
			
			input_data[input_data_pos]='\0';
			
			if(input_data_pos>=sizeof(input_data)-1)
			{
				fprintf(stdout,"%s:%d Character overload\n",__FILE__,__LINE__);
				break;
			}
		}
		
		input_data[input_data_pos]='\0';
		input_data_pos=0;
		
		fprintf(stdout,"%s:%d Got input string = [%s]\n",__FILE__,__LINE__,input_data);
		
		set_value_from_string(input_data,data);
		
		char validate[7+1]={0};
		get_string_from_value(data,validate,7,INVERT);
		
		int ndata = sizeof(float) * 7 * TRANSLATION_TBL_LEN; // This is tricky, it number of bytes not number of element

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
		
		void *buff = TF_TensorData(output_values[0]);
		int64_t num_elements = TF_TensorElementCount(output_values[0]);
		
		float *offsets = (float *)buff;
		
		printf("Input Tensor %lld:\n",ndata/sizeof(float));
		print_tensor(data,ndata/sizeof(float));
		
		printf("Result Tensor %lld:\n",num_elements);
		print_tensor(offsets,num_elements);
		
		//need '\0' in the end
		char output_s[4+1]={0};
		
		get_string_from_value(offsets,output_s,4,0);
		
		fprintf(stdout,"%s:%d INPUT VALUE [%s] ACTUAL [\"%s\"] = [\"%s\"] (Calculated \"true\" result=%ld)\n",__FILE__,__LINE__,input_data,validate,output_s,get_actual_result(input_data));
		
		TF_DeleteTensor(int_tensor);
		TF_DeleteTensor(output_values[0]);
		
	quit_prog: ;
	}
	
	// //Free memory
	TF_DeleteGraph(graph);
	TF_DeleteSession(session, status);
	TF_DeleteSessionOptions(session_opts);
	TF_DeleteStatus(status);
}

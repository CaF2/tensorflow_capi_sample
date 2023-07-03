#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void *data, size_t a, void *b)
{
}

int main(const int argc, char **const argv)
{
	//********* Read model
	TF_Graph *Graph = TF_NewGraph();
	TF_Status *Status = TF_NewStatus();

	TF_SessionOptions *SessionOpts = TF_NewSessionOptions();
	TF_Buffer *RunOpts = NULL;

	const char *saved_model_dir = "lstm2/";
	const char *tags = "serve"; // default model serving tag; can change in future
	int ntags = 1;

	TF_Session *Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
	if (TF_GetCode(Status) == TF_OK)
	{
		fprintf(stdout,"%s:%d TF_LoadSessionFromSavedModel [OK]\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stdout,"%s:%d TF_LoadSessionFromSavedModel [%s]\n",__FILE__,__LINE__,TF_Message(Status));
	}

	//****** Get input tensor
	// TODO : need to use saved_model_cli to read saved_model arch
	int NumInputs = 1;
	TF_Output Input[NumInputs];
	
	//Check also TF_GraphNextOperation()
	TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
	if (t0.oper == NULL)
	{
		fprintf(stderr,"%s:%d ERROR: Failed TF_GraphOperationByName serving_default_input_1\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stdout,"%s:%d TF_GraphOperationByName serving_default_input_1 is OK\n",__FILE__,__LINE__);
	}

	Input[0] = t0;

	//********* Get Output tensor
	int NumOutputs = 1;
	TF_Output Output[NumOutputs];
	
	//Check also TF_GraphNextOperation()
	TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
	if (t2.oper == NULL)
	{
		fprintf(stderr,"%s:%d ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stdout,"%s:%d TF_GraphOperationByName StatefulPartitionedCall is OK\n",__FILE__,__LINE__);
	}

	Output[0] = t2;

	//********* Allocate data for inputs & outputs
	TF_Tensor *InputValues[NumInputs];
	TF_Tensor *OutputValues[NumOutputs];

	int ndims = 2;
	int64_t dims[] = {1, 30};
	float data[1 * 30]; //= {1,1,1,1,1,1,1,1,1,1};
	for (int i = 0; i < (1 * 30); i++)
	{
		data[i] = 1.00;
	}
	int ndata = sizeof(float) * 1 * 30; // This is tricky, it number of bytes not number of element

	TF_Tensor *int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
	if (int_tensor != NULL)
	{
		fprintf(stdout,"%s:%d TF_NewTensor is OK\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stderr,"%s:%d ERROR: Failed TF_NewTensor\n",__FILE__,__LINE__);
	}

	InputValues[0] = int_tensor;

	// //Run the Session
	TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

	if (TF_GetCode(Status) == TF_OK)
	{
		fprintf(stdout,"%s:%d Session is OK\n",__FILE__,__LINE__);
	}
	else
	{
		fprintf(stderr,"%s:%d Session returned with [%s]\n",__FILE__,__LINE__, TF_Message(Status));
	}

	// //Free memory
	TF_DeleteGraph(Graph);
	TF_DeleteSession(Session, Status);
	TF_DeleteSessionOptions(SessionOpts);
	TF_DeleteStatus(Status);

	void *buff = TF_TensorData(OutputValues[0]);
	int64_t num_elements = TF_TensorElementCount(OutputValues[0]);
	
	float *offsets = (float *)buff;
	printf("Result Tensor %lld:\n",num_elements);
	//for (int i = 0; i < num_elements; i++)
	for (int i = 0; i < 10; i++)
	{
		printf("%f\n", offsets[i]);
	}
	
	TF_DeleteTensor(int_tensor);
	TF_DeleteTensor(OutputValues[0]);
}

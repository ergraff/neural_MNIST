/*
 * neural_numbers.c
 *
 *  Created on: 21 feb. 2019
 *      Author: Erik Graff
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>
#include <strings.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <direct.h>


double sigmoid(double x);
double inverseSigmoid(double x);
double inputSigmoid(double x);
double costFunction(void);
void randomizeLayer(double* x);
void randomizeWeight(double* weight, int rows, int columns);
void calculateOutput(void);
void randomizeInput(void);
void printLayersAndWeights(void);
void printResult(void);
void saveNeuroToFile(void);
void readNeuroFromFile(void);
void printImage(void);
void backPropagation(void);
double meanValue(double* layer, int count);

double train_meanValue(int row, int column);
void train_image(void);

//Lazy global vars
double inputLayer[784] = {0};
double firstLayer[16] = {0}, secondLayer[16] = {0}, outputLayer[10] = {0};
double inputWeight[784][16] = {0}, firstWeight[16][16] = {0}, outputWeight[16][10] = {0};
double firstLayerBias = -7;
double secondLayerBias = -4;
double outputLayerBias = -5;

//Train
double train_weight_average[816][16][100] = {0};
const unsigned int train_image_start = 16;
const unsigned int train_image_size = 784;
const unsigned int train_label_start = 8;
unsigned int train_image_counter = 0;
unsigned int train_image_label = 0;
unsigned int train_backPropagation_counter = 0;


int main(void)
{
	srand(time(NULL));

	//Randomize weights
	/*
	randomizeWeight(&inputWeight[0][0],784,16);
	randomizeWeight(&firstWeight[0][0],16,16);
	randomizeWeight(&outputWeight[0][0],16,10);
	*/
	//Save weights to .neuro file
	//saveNeuroToFile();

	//Read weights from .neuro file and save them for use in their arrays
	readNeuroFromFile();

	//Randomize input
	//randomizeInput();

	//Calculate output
	//calculateOutput();

	//Print layers and weights
	//printLayersAndWeights();

	//Print the result
	//printResult();

	//Adjust with back propagation
	//backPropagation();

	//Recalculate output
	//calculateOutput();

	//FUCKIN' TRAIN CHO CHO FINALLY
	train_image();

	//Print layers and weights, and result, anew
	printLayersAndWeights();
	printResult();

	//Print the image input
	printImage();

	getchar();

	return 0;
}

void train_image(void)
{
	FILE *readImage = fopen("images.idx3-ubyte","rb");
	FILE *readLabel = fopen("labels.idx1-ubyte","rb");

	//Train with 60 000 images, test with 1000
	for(unsigned int i = 0; i < 1000; i++) //Train with the set of 60000 images
	{
		fseek(readImage,(train_image_start+train_image_size*train_image_counter),SEEK_SET);
		fseek(readLabel,(train_label_start+train_image_counter),SEEK_SET);
		fread(inputLayer,1,784,readImage);
		fread(&train_image_label,1,1,readLabel);

		for(int j = 0; j < 784; j++) //Sigmoid the input 0-1
		{
			inputLayer[j] = inputSigmoid((double)inputLayer[j]);
		}

		train_image_counter++;

		calculateOutput(); 	//Calculate
		backPropagation();	//Adjust

		if(train_image_counter % 100 == 0)
		{
			printf("%i/60000 images processed. %.1lf%% done.\n",train_image_counter,train_image_counter/600.0);
		}
		else if(train_image_counter == 60000)
		{
			printf("All images processed. Network saved to file.\n");
		}
	}

	//Check train_weight_average
	/*
	for(int i = 0; i < 816; i++)
	{
		for(int j = 0; j < 16; j++)
		{
			for(int k = 0; k < 100; k++)
			{
				printf("%lf\n",train_weight_average[i][j][k]);
			}
		}
	}
	*/


	saveNeuroToFile();

	fclose(readImage);
	fclose(readLabel);
}

void backPropagation(void)
{
	for(int i = 0; i < 10; i++) //Adjust weights to output
	{
		for(int j = 0; j < 16; j++)
		{
			if(i == train_image_label) //Weights for right output
			{
				if(outputWeight[j][i] > 0) //Increase positive weights
				{
					train_weight_average[800+j][i][train_backPropagation_counter] = secondLayer[j]/3.0;
					//outputWeight[j][i] = sigmoid(inverseSigmoid(outputWeight[j][i])*1.01);
				}
				else if(outputWeight[j][i] < 0) //Turn negative weights positive
				{
					train_weight_average[800+j][i][train_backPropagation_counter] = secondLayer[j]/3.0;
					//outputWeight[j][i] = -1*sigmoid(inverseSigmoid(outputWeight[j][i]*-1)/1.01);
				}
			}
			else //Weights for wrong output
			{
				if(outputWeight[j][i] > 0) //Decrease positive weights
				{
					train_weight_average[800+j][i][train_backPropagation_counter] = secondLayer[j]*-1/3.0;
					//outputWeight[j][i] = sigmoid(inverseSigmoid(outputWeight[j][i])/1.01);
				}
				else if(outputWeight[j][i] < 0) //Turn negative weights positive
				{
					train_weight_average[800+j][i][train_backPropagation_counter] = secondLayer[j]/3.0;
					//outputWeight[j][i] = -1*sigmoid(inverseSigmoid(outputWeight[j][i]*-1)*1.01);
				}
			}
		}
	}

	for(int i = 0; i < 16; i++) //Adjust weights to secondLayer
	{
		for(int j = 0; j < 16; j++)
		{
			if(secondLayer[i] > meanValue(&secondLayer[0],16)/2) //Weights for strong neurons (correct)
			{
				if(firstWeight[j][i] > 0) //Increase positive weights
				{
					train_weight_average[784+j][i][train_backPropagation_counter] = firstLayer[j]/3.0;
					//firstWeight[j][i] = sigmoid(inverseSigmoid(firstWeight[j][i])*1.0001);
				}
				else if(firstWeight[j][i] < 0) //Turn negative weights positive
				{
					train_weight_average[784+j][i][train_backPropagation_counter] = firstLayer[j]/3.0;
					//firstWeight[j][i] = -1*sigmoid(inverseSigmoid(firstWeight[j][i]*-1)/1.0001);
				}
			}
			else //Weights for weak neurons (wrong)
			{
				if(firstWeight[j][i] > 0) //Decrease positive weights
				{
					train_weight_average[784+j][i][train_backPropagation_counter] = firstLayer[j]*-1/3.0;
					//firstWeight[j][i] = sigmoid(inverseSigmoid(firstWeight[j][i])/1.0001);
				}
				else if(firstWeight[j][i] < 0) //Increase negative weights
				{
					train_weight_average[784+j][i][train_backPropagation_counter] = firstLayer[j]*-1/3.0;
					//firstWeight[j][i] = -1*sigmoid(inverseSigmoid(firstWeight[j][i]*-1)*1.0001);
				}
			}
		}
	}

	for(int i = 0; i < 16; i++) //Adjust weights to firstLayer
	{
		for(int j = 0; j < 784; j++)
		{
			if(firstLayer[i] > meanValue(&firstLayer[0],16)/2) //Weights for strong neurons (correct)
			{
				if(inputWeight[j][i] > 0) //Increase positive weights
				{
					train_weight_average[j][i][train_backPropagation_counter] = inputLayer[j];
					//inputWeight[j][i] = sigmoid(inverseSigmoid(inputWeight[j][i])*1.0001);
				}
				else if(firstWeight[j][i] < 0) //Turn negative weights positive
				{
					train_weight_average[j][i][train_backPropagation_counter] = inputLayer[j];
					//inputWeight[j][i] = -1*sigmoid(inverseSigmoid(inputWeight[j][i]*-1)/1.0001);
				}
			}
			else //Weights for weak neurons (wrong)
			{
				if(inputWeight[j][i] > 0) //Decrease positive weights
				{
					train_weight_average[j][i][train_backPropagation_counter] = -1.0*inputLayer[j];
					//inputWeight[j][i] = sigmoid(inverseSigmoid(inputWeight[j][i])/1.0001);
				}
				else if(inputWeight[j][i] < 0) //Increase negative weights
				{
					train_weight_average[j][i][train_backPropagation_counter] = -1.0*inputLayer[j];
					//inputWeight[j][i] = -1*sigmoid(inverseSigmoid(inputWeight[j][i]*-1)*1.0001);
				}
			}
		}
	}

	train_backPropagation_counter++;

	if(train_backPropagation_counter == 99) //Backpropagated 100 times?
	{
		printf("Applying changes to weights.\n");
		printf("inputWeight.\n"); //Felsökning
		for(int i = 0; i < 784; i++) //Adjust weights of inputWeight with the mean value of adjustments
		{
			for(int j = 0; j < 16; j++)
			{
				inputWeight[i][j] += train_meanValue(i,j);
				printf("%lf\n",train_meanValue(i,j)); //Felsökning
			}
		}
		printf("firstWeight.\n"); //Felsökning
		for(int i = 784; i < 800; i++) //Adjust weights of firstWeight with the mean value of adjustments
		{
			for(int j = 0; j < 16; j++)
			{
				firstWeight[i-784][j] += train_meanValue(i,j);
				printf("%lf\n",train_meanValue(i,j)); //Felsökning
			}
		}
		printf("outputWeight.\n"); //Felsökning
		for(int i = 800; i < 816; i++) //Adjust weights of outputWeight with the mean value of adjustments
		{
			for(int j = 0; j < 10; j++)
			{
				outputWeight[i-800][j] += train_meanValue(i,j);
				printf("%lf\n",train_meanValue(i,j)); //Felsökning
			}
		}

		train_backPropagation_counter = 0;
	 }
}

double train_meanValue(int row, int column)
{
	double sum = 0;
	for(int i = 0; i < 100; i++)
	{
		sum += train_weight_average[row][column][i];
	}
	return sum/100.0;
}

double meanValue(double* layer, int count)
{
	double sum = 0;

	for(int i = 0; i < count; i++)
	{
		sum += *layer;
		layer++;
	}

	return sum/((double) count);
}

void printImage(void)
{
	int counter = 0;
	printf("\n");
	for(int i = 0; i < 28; i++)
	{
		for(int j = 0; j < 28; j++)
		{
			if(inputLayer[counter] < 0.5)
			{
				printf("  ");
			}
			else
			{
				printf("# ");
			}
			counter++;
		}
		printf("\n");
	}
}

void saveNeuroToFile(void)
{
	FILE *write = fopen("network.neuro","wb");

	fwrite(inputWeight,sizeof(double),12544,write);
	fwrite(firstWeight,sizeof(double),256,write);
	fwrite(outputWeight,sizeof(double),160,write);

	fclose(write);
}

void readNeuroFromFile(void)
{
	FILE *read = fopen("network.neuro","rb");

	fread(inputWeight,sizeof(double),12544,read);
	fread(firstWeight,sizeof(double),256,read);
	fread(outputWeight,sizeof(double),160,read);

	fclose(read);
}

double costFunction(void)
{
	double cost = 0;
	for(int i = 0; i < 10; i++)
	{
		if(i == train_image_label)
		{
			cost += (outputLayer[i]-1)*(outputLayer[i]-1);
		}
		else
		{
			cost += outputLayer[i]*outputLayer[i];
		}
	}
	return cost;
}

void printLayersAndWeights(void)
{
	printf("\nFirst 20 elements of inputLayer:\n");
	for(int i = 0; i < 20; i++)
	{
		printf("%lf\n",inputLayer[i]);
	}
	printf("\nFirst layer:\n");
	for(int i = 0; i < 16; i++)
	{
		printf("%lf\n",firstLayer[i]);
	}
	printf("\nSecond layer:\n");
	for(int i = 0; i < 16; i++)
	{
		printf("%lf\n",secondLayer[i]);
	}
	printf("\nFirst 5 rows of inputWeight:\n");
	for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 16; j++)
		{
			printf("%lf ",inputWeight[i][j]);
		}
		printf("\n");
	}
	printf("\nFirst 5 rows of firstWeight:\n");
	for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 16; j++)
		{
			printf("%lf ",firstWeight[i][j]);
		}
		printf("\n");
	}
	printf("\nFirst 5 rows of outputWeight:\n");
	for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 16; j++)
		{
			printf("%lf ",outputWeight[i][j]);
		}
		printf("\n");
	}
	printf("\noutputLayer:\n");
	for(int i = 0; i < 10; i++)
	{
		printf("%lf\n",outputLayer[i]);
	}
}

void printResult(void)
{
	int result = 0;
	double largestResult = outputLayer[0];
	for(int i = 1; i < 10; i++)
	{
		if(outputLayer[i] > largestResult)
		{
			result = i;
			largestResult = outputLayer[i];
		}
	}
	printf("The neural network thinks it's: %i. Cost = %lf (label = %i)\n",result,costFunction(),train_image_label);
}

void calculateOutput(void)
{
	double sum;					//Calculate firstLayer
		for(int i = 0; i < 16; i++)
		{
			sum = 0;
			for(int j = 0; j < 784; j++)
			{
				sum += (double)inputLayer[j]*inputWeight[j][i];
			}
			firstLayer[i] = sigmoid(sum - firstLayerBias);
		}

		for(int i = 0; i < 16; i++) //Calculate secondLayer
		{
			sum = 0;
			for(int j = 0; j < 16; j++)
			{
				sum += firstLayer[j]*firstWeight[j][i];
			}
			secondLayer[i] = sigmoid(sum - secondLayerBias);
		}

		for(int i = 0; i < 10; i++) //Calculate outputLayer
		{
			sum = 0;
			for(int j = 0; j < 16; j++)
			{
				sum += secondLayer[j]*outputWeight[j][i];
			}
			outputLayer[i] = sigmoid(sum - outputLayerBias)/3;
		}
}

void randomizeInput(void)
{
	for(int i = 0; i < 784; i++)
	{
		inputLayer[i] = inputSigmoid((double) (rand() % 256));
	}
}

void randomizeWeight(double* weight, int rows, int columns)
{
	double (*weightPtr)[columns] = weight, r;

	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < columns; j++)
		{
			r = sigmoid((double) (rand() % 21));
			if(rand() % 2)
			{
				r *= -1;
			}
			weightPtr[i][j] = r;
		}
	}

}

void randomizeLayer(double* layerPtr)
{
	for(int i = 0; i < 16; i++)
	{
		*layerPtr = sigmoid((double) (rand() % 21));
		layerPtr++;
	}
}

double inverseSigmoid(double x)
{
	return 10-2*log(3/x-1);
}

double sigmoid(double x)
{
	return 3/(1+exp(-0.5*x+5));
}

double inputSigmoid(double x)
{
	return 1/(exp(-0.05*x+6)+1);
}

/*
 *  A multilayer perceptron,only support logstic 
 *  sigmod,and one output.
 *  You must chang it yourself if you need.
 *  I put every part of the net in heap.
 *  So you can build a very large net.
 *  Though it's not recommanded
 *                                 by GuoHaotian
 *              Email: minority1728645@gmail.com
 */

#include <stdlib.h>
#include <math.h>

#define alpha 1.0
#define beta 1.0
#define eta 0.2

//sigmod add the function you needed
float logstic(float x)
{
    return 1.0/(1.0+exp(-(alpha*x)));
}

float derivative_logstic(float y)
{
    return alpha*y*(1-y);
}

float sigtanh(float x)
{
    return alpha*tanh(beta*x);
}

float derivative_sigtanh(float y)
{
    return (1.0*beta/alpha)*(alpha-y)*(alpha+y);
}

typedef struct _sigmod
{
    float (*origin)(float);
    float (*derivative)(float);
} st_sigmod;

typedef struct _feature
{
    int index;
    float attr;
} feature;

typedef struct _sample
{
    int num_target;
    int target;
    int num_features;
    feature *features;
} sample;

typedef struct _neuron
{
	float output;
    float *weights;
    float delta;
} neuron;

typedef struct _layer
{
    int num_neurons;
    neuron *neurons;
    st_sigmod sigmod;
} layer;

typedef struct _net
{
    int num_layers;
    layer *layers;
} net;

void forward(layer *former,layer *latter)
{
    int i,j;
    for(i=0;i<latter->num_neurons;i++)
    {
        float out=0;
        for(j=0;j<former->num_neurons;j++)
        {
            out+=((former->neurons)[j].output)*((latter->neurons)[i].weights[j]);
        }
        (latter->neurons)[i].output=latter->sigmod.origin(out);
    }
}

void backward(layer *former,layer *latter)
{
    int i,j;
    for(i=0;i<former->num_neurons;i++)
    {
        float y=(former->neurons)[i].output;
        float sum=0;
        for(j=0;j<latter->num_neurons;j++)
        {
            sum+=((latter->neurons)[j].delta) * ((latter->neurons)[j].weights[i]);
        }
        (former->neurons)[i].delta=former->sigmod.derivative(y)*sum;
    }
}


void updateweight(layer *former,layer *latter)
{
    int i,j;
    for(i=0;i<latter->num_neurons;i++)
    {
        for(j=0;j<former->num_neurons;j++)
        {
            ((latter->neurons)[i].weights[j])+=eta*((latter->neurons)[i].delta)*((former->neurons)[j].output);
        }
        
    }
}

void set_sigmod(layer *l,char sig)
{
    switch(sig){
        default :
        case 'l':
            l->sigmod.origin=logstic;
            l->sigmod.derivative=derivative_logstic;
            break;

        case 't':
            l->sigmod.origin=sigtanh;
            l->sigmod.derivative=derivative_sigtanh;
            break;
    }
}

void init_input_layer(layer *l,int num_neurons,char sig)
{
    l->num_neurons=num_neurons; 
    l->neurons=(neuron*)malloc(sizeof(neuron)*num_neurons);
    set_sigmod(l,sig);
}

void free_input_layer(layer *l,int num_neurons)
{
    free(l->neurons);
}

//Set all input zero.
void clear_layer(layer *l)
{
    int i;
    for(i=0;i<(l->num_neurons);i++)
    {
        (l->neurons)[i].output=0;
    }
}

void init_layer(layer *l,int num_inputs,int num_neurons,char sig)
{
    int i,j;
    l->num_neurons=num_neurons; 
    l->neurons=(neuron*)malloc(sizeof(neuron)*num_neurons);
    set_sigmod(l,sig);
    for(i=0;i<num_neurons;i++)
    {
        (l->neurons)[i].weights=(float *)malloc(sizeof(float)*num_inputs);
        for(j=0;j<num_inputs;j++)
        {
            (l->neurons)[i].weights[j]=rand()/(float)(RAND_MAX)*2-1;
        }
        
    }
}

void free_layer(layer *l,int num_inputs,int num_neurons)
{
    int i;
    for(i=0;i<num_neurons;i++)
    {
        free((l->neurons)[i].weights);
    }
    free(l->neurons);
}

void init_net(net *n,int num_layers,int *num_neurons,char *sigmods)
{
    int i=0;
    n->num_layers=num_layers;
    n->layers=(layer *)malloc(sizeof(layer)*num_layers);

    init_input_layer(&(n->layers[0]),num_neurons[0],sigmods[0]);

    for(i=1;i<num_layers;i++)
        init_layer(&(n->layers[i]),n->layers[i-1].num_neurons,num_neurons[i],sigmods[i]);
}

void free_net(net *n,int num_layers,int *num_neurons)
{
    int i=0;

    free_input_layer(&(n->layers[0]),num_neurons[0]);

    for(i=1;i<num_layers;i++)
        free_layer(&(n->layers[i]),n->layers[i-1].num_neurons,num_neurons[i]);
    
    n->layers=(layer *)malloc(sizeof(layer)*num_layers);
}

void train(net *n,int num_samples,sample *samples,int epochs)
{
    int e,i,j;
    for(e=0;e<epochs;e++)
    {
        for(i=0;i<num_samples;i++)
        {
            clear_layer(&(n->layers[0]));
            for(j=0;j<(samples[i].num_features);j++)
            {
                n->layers[0].neurons[samples[i].features[j].index-1].output=samples[i].features[j].attr;
            }

            for(j=1;j<n->num_layers;j++)
            {
                forward(&(n->layers[j-1]),&(n->layers[j]));
            }

            for(j=0;j<samples[i].num_target;j++)
            {
                float oj=n->layers[n->num_layers-1].neurons[j].output;
                n->layers[n->num_layers-1].neurons[j].delta=n->layers[n->num_layers-1].sigmod.derivative(oj)*(samples[i].target-oj);
            }

            for(j=n->num_layers-1;j>0;j--)
            {
                backward(&(n->layers[j-1]),&(n->layers[j]));
            }

            for(j=1;j<n->num_layers;j++)
            {
                updateweight(&(n->layers[j-1]),&(n->layers[j]));
            }
        }
    }
}

void predict(net *n,int num_test,sample* test_samples,float *scores)
{
    int i,j;
    for(i=0;i<num_test;i++)
    {
        clear_layer(&(n->layers[0]));
        for(j=0;j<(test_samples[i].num_features);j++)
        {
            n->layers[0].neurons[test_samples[i].features[j].index-1].output=test_samples[i].features[j].attr;
        }

        for(j=1;j<n->num_layers;j++)
        {
            forward(&(n->layers[j-1]),&(n->layers[j]));
        }
        for(j=0;j<1;j++)
        {
            scores[i]=n->layers[n->num_layers-1].neurons[j].output;
        }
    }
}






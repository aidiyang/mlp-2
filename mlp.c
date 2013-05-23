/*
 *  A multilayer perceptron.
 *  Support the input format:
 *  label dim:value dim:value ..... 
 *  Compatible with liblinear.
 *                                  by GuoHaotian
 *               Email: minority1728645@gmail.com
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mlp.h"	

#define MAX_LINE_LEN 4096

#define EPOCHS 10

char* readline(FILE *input,char* line)
{
    int len;

    int max_line_len=MAX_LINE_LEN;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

int count_lines(char *filename) 
{
    FILE *file=fopen(filename, "r");
    int count=0;
    char buf[4096];
    while(fgets(buf,4096,file) != NULL)
    {
        if(strrchr(buf,'\n') != NULL)
            count++;
    }
    return count;
}

int count_char(char *str,char c)
{
    int i;
    int count=0;
    for(i=0;i<strlen(str);i++)
        if(c==str[i])
            count++;
    return count;
}

void read_samples(char* filename,sample *samples,int num_samples)
{
    int i,j;

    char *line=(char*)malloc(sizeof(char)*MAX_LINE_LEN);

    FILE *sample_file=fopen(filename,"r");

    for(i=0;i<num_samples;i++)
    {
        line=readline(sample_file,line);
        int num_f=count_char(line,':');
        samples[i].num_features=num_f;
        samples[i].num_target=1;
        samples[i].target=atoi(strtok(line," "));
        samples[i].features=(feature*)malloc(sizeof(feature)*num_f);
        for(j=0;j<num_f;j++)
        {
            sscanf(strtok(NULL," "),"%d:%f",&(samples[i].features[j].index),&(samples[i].features[j].attr));
        }
    }
    
    fclose(sample_file);
}

void free_samples(sample *samples,int num_samples)
{
    int i;
    for(i=0;i<num_samples;i++)
        free(samples[i].features);

    free(samples);
}

int main(int argc,char *argv[]){
    int i=0;
    srand(time(NULL));

    net mlpnet;
    int layer_neurons[]={5000,10,5,1};
    init_net(&mlpnet,4,layer_neurons);
    
    float *inputs=(float*)malloc(sizeof(float)*2);

    sample *samples;

    int num_lines=count_lines(argv[1]);

    samples=(sample*)malloc(sizeof(sample)*num_lines);

    read_samples(argv[1],samples,num_lines);

    train(&mlpnet,num_lines,samples,EPOCHS);
    
    free_samples(samples,num_lines);

    //==================================================

    num_lines=count_lines(argv[2]);
    sample *test_samples=(sample*)malloc(sizeof(sample)*num_lines);

    read_samples(argv[2],test_samples,num_lines);

    float *scores=(float*)malloc(sizeof(float)*num_lines);
    predict(&mlpnet,num_lines,test_samples,scores);

    free_samples(test_samples,num_lines);

    free_net(&mlpnet,4,layer_neurons);

    char scorefile[100];
    sprintf(scorefile,"%s.score",argv[1]);
    FILE *score=fopen(scorefile,"w");

    for(i=0;i<num_lines;i++)
    {
        fprintf(score,"%f\n",scores[i]);
    }

    fclose(score);

    return 0;
}






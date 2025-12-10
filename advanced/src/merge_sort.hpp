#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <algorithm>
#include <utility>

#include <mpi.h>

#define ROOT 0

struct Path_data{
    double val;
    double global_idx;
    double cashflow;
    double exercise_idx;
    double worker;
};


void merge_sort
(
    std::vector<Path_data> input, 
    int level, 
    int pid, 
    std::vector<double>& buff, 
    int num_levels,
    std::vector<MPI_Request>& recv_request, 
    std::vector<MPI_Status>& recv_status,
    int max_paths_per_proc,
    int N,
    int nproc
){

    int parent = pid & ~(1 << level);
    int child = pid | (1<<(level-1));


    std::vector<Path_data> result;

    if(!level){
        std::sort(input.begin(), input.end(), [](const Path_data& a, const Path_data& b) {
            return a.val < b.val;
        });

        result = input;
        
    } else{
        //recv from child
        MPI_Wait(&(recv_request[level-1]), &(recv_status[level-1]));

        int count;
        MPI_Get_count(&recv_status[level-1], MPI_DOUBLE, &count);

        std::vector<Path_data> child_input(count / 5);

        int MS_msg_size = max_paths_per_proc * 5;
        int offset = max_paths_per_proc * 5;

        for(int i = 1; i < level; i++){
            offset += MS_msg_size;
            MS_msg_size *= 2;
        }

        for(int i = 0; i < static_cast<int>(child_input.size()); i++){
            double val = buff[offset];
            double global_idx = buff[offset+1];
            double cashflow = buff[offset+2];
            double exercise_idx = buff[offset+3];
            double worker = buff[offset+4];

            child_input[i] = Path_data{val, global_idx, cashflow, exercise_idx, worker};

            offset += 5;
        }


        //merge with child
        result.resize(input.size() + child_input.size());
        std::merge(input.begin(), input.end(), child_input.begin(), child_input.end(), result.begin(), 
            [](const Path_data& a, const Path_data& b) {
                return a.val < b.val;
        });
    }

    if((level == (num_levels - 1)) || (parent != pid)){
        //put sorted vec into buff
        for(int i = 0; i < static_cast<int>(result.size()); i++){
            buff[5 * i] = result[i].val;
            buff[5 * i+1] = result[i].global_idx;
            buff[5 * i+2] = result[i].cashflow;
            buff[5 * i+3] = result[i].exercise_idx;
            buff[5 * i+4] = result[i].worker;
        }
    }

    //send to parent
    if(level != (num_levels - 1)){
        if(pid != parent){
            MPI_Send(&buff[0], result.size() * 5, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD);
        } else{
            merge_sort(result, level+1, pid, buff, num_levels, recv_request, recv_status, max_paths_per_proc, N, nproc);
        }
    }
}
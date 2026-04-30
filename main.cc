#include <math.h>
#include "PRNG.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>



//Function definition for monte carlo simulation
double monteCarloAsianSerial(int samples, double St0, double strike, double sigma, double r, double T);

double monteCarloAsianParallel(int samples, double St0, double strike, double sigma, double r, double T);

int main(){
    //Example inputs
    int samples = 1000000;       //Number of simulation paths (higher = more accurate, but slower)
    double St0 = 100.0;          //Initial stock price (S_0)
    double strike = 103.0;       //Strike price (K)
    double sigma = 0.10;         //Volatility
    double r = 0.01;             //Risk-free interest rate 
    double T = 255.0;


    std::cout << "Running Monte Carlo simulation" << std::endl;
    std::cout << "Samples: " << samples << std::endl;

    //Call the function and store the result
    double startAsianSerial = omp_get_wtime();
    double estimatedAsianPriceSerial = monteCarloAsianSerial(samples, St0, strike, sigma, r, T);
    double endAsianSerial = omp_get_wtime();
    //double valueDiffAsianSerial = abs(expectedValue - estimatedPriceSerial);
    double serialAsianTime = endAsianSerial - startAsianSerial;

    double startAsianParallel = omp_get_wtime();
    double estimatedAsianPriceParallel = monteCarloAsianSerial(samples, St0, strike, sigma, r, T);
    double endAsianParallel = omp_get_wtime();
    //double valueDiffAsianSerial = abs(expectedValue - estimatedPriceSerial);
    double ParallelAsianTime = endAsianParallel - startAsianParallel;

    

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Estimated Option Price (Serial): €" << estimatedAsianPriceSerial << " Took: " << serialAsianTime << " seconds. Difference from expected value = " << std::endl;
    std::cout << "Estimated Option Price (Parallel): €" << estimatedAsianPriceParallel << " Took: " << ParallelAsianTime << " seconds. Difference from expected value = " << std::endl;
   
    return 0;
}


//Serial Monte Carlo implementation.
double monteCarloAsianSerial(int samples, double St0, double strike, double sigma, double r, double T){
    double dt = 1.0 / 255.0;
    double mu_t = (r - 0.5 * sigma * sigma) * dt;
    double step_vol = sigma * sqrt(dt);
    double total_payoff = 0.0;

    PRNG myGenerator(1);

    //Loop for each path
    for(int i = 0; i < samples; ++i){
        double current_S = St0;
        double sum_S = St0; //Include starting price in the running sum
        
        //Loop through individual days on each path
        for(int t = 1; t <= T; ++t){
            double randVariate = myGenerator.getStandardNormal();

            //Calculate current days value
            current_S *= exp(mu_t + step_vol * randVariate);

            
            sum_S += current_S;//Sum to total
        }

        //Caculate average for 255+1 as includes final day so 0-> 255 days
        double A_T = sum_S / (T + 1.0);

        //Caculate path payoff.
        double path_payoff = std::max(A_T - strike, 0.0);
        total_payoff += path_payoff;
    }
    

    total_payoff = total_payoff/samples;

    //Could assume year is just 1 if always using 255 days but here to allow change
    double T_years = T / 255.0; 
    double payoff_value = exp(-r * T_years) * total_payoff;

    return payoff_value;
}

double monteCarloAsianParallel(int samples, double St0, double strike, double sigma, double r, double T) {
    double dt = 1.0 / 255.0;
    double mu_t = (r - 0.5 * sigma * sigma) * dt;
    double step_vol = sigma * sqrt(dt);
    double total_payoff = 0.0;


    //Use reduction to add up total_payoff across all threads
    #pragma omp parallel for reduction(+:total_payoff)
    for(int i = 0; i < samples; ++i) {
        
        //Seed with i for each thread to have its own PRNG
        PRNG myGenerator(1 + i); 

        double current_S = St0;
        double sum_S = St0; //Include starting price in the running sum
        
        //Loop through individual days on each path
        for(int t = 1; t <= T; ++t) {
            double randVariate = myGenerator.getStandardNormal();

            // Calculate current day's value
            current_S *= exp(mu_t + step_vol * randVariate);
            sum_S += current_S; // Sum to total
        }

        //Caculate average for 255+1 as includes final day so 0-> 255 days
        double A_T = sum_S / (T + 1.0);

        //Calculate path payoff
        double path_payoff = std::max(A_T - strike, 0.0);
        

        total_payoff += path_payoff;
    }
    
    total_payoff = total_payoff / samples;

    //Discount back to present value
    double T_years = T / 255.0; 
    double payoff_value = exp(-r * T_years) * total_payoff;

    return payoff_value;
}


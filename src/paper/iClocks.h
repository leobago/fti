/** @author Jorge AMAYA (jorgeluis.amaya@gmail.com)
 *  @license GPL-3.0 <https://opensource.org/licenses/GPL-3.0>
 *
 *   Copyright (c) 2016 KU Leuven University
 *   Some rights reserved. See COPYING, AUTHORS.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __CLOCKS_H__
#define __CLOCKS_H__

#include <chrono>

typedef enum CLOCK_ID {
    ITER,
    CKPT,
    RECO,
    NUM_CLOCK_IDS
} CLOCK_ID;

class Clocks{
  public:
    Clocks() {
        t0 = std::chrono::system_clock::now();
        dt = new std::chrono::duration<double>[NUM_CLOCK_IDS];
        t  = new std::chrono::duration<double>[NUM_CLOCK_IDS];
        fill( t, t+NUM_CLOCK_IDS, 0 );
    }
    
    ~Clocks() {
        delete[] dt;
        delete[] t;
    };

    inline void start(const int i) { 
        dt[i] = std::chrono::system_clock::now(); 
    }
    
    inline void stop (const int i) {
      dt[i] = std::chrono::system_clock::now() - dt[i];
      t [i]+= dt[i];
    }
    
    inline double get_t(const int i) {
        return t[i].count(); 
    }
    
    inline double get_dt(const int i) {
        return dt[i].count(); 
    }

    inline double get_dT() { 
        auto tp = std::chrono::system_clock::now();
        return  (tp - t0).count(); 
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> t0;
    std::chrono::duration<double> *dt;
    std::chrono::duration<double> *t;
};

#endif

#ifndef CUEST_JK_GRAD_H
#define CUEST_JK_GRAD_H

#include "jk_grad.h"

#ifdef USING_cuEST
#include <cuest.h>

namespace psi {
namespace scfgrad {

class cuESTJKGrad : public JKGrad {
   protected:
    std::shared_ptr<BasisSet> auxiliary_;
    std::shared_ptr<MintsHelper> mints_;
    double condition_;

   public:
    cuESTJKGrad(int deriv, std::shared_ptr<MintsHelper> mints);
    ~cuESTJKGrad() override;

    void compute_gradient() override;
    void compute_hessian() override;
    void print_header() const override;

    void set_condition(double condition) { condition_ = condition; }
};

}  // namespace scfgrad
}  // namespace psi

#endif  // USING_cuEST
#endif  // CUEST_JK_GRAD_H

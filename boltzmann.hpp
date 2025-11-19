#include <Eigen/Dense>
#include <random>
#include <cmath>


inline double sigmoid(const double& x) {
    return 1 / (1 + std::exp(-x)); 
}

inline Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
    return x.unaryExpr([](double x_i) { return sigmoid(x_i); }); 
}


// Implementation of a Bernoulli-Bernoulli Restricted Boltzamann Machine
class BernoulliRBM {

    size_t n_visible, n_hidden; 
    Eigen::VectorXd v_bias, h_bias;
    Eigen::MatrixXd weights; 
    std::mt19937 rng; 

public:

    BernoulliRBM(size_t n_visible, size_t n_hidden) : 
        n_visible{n_visible}, n_hidden{n_hidden}, 
        rng{std::random_device{}()} {
        this->init_params(); 
    }


    void train(const std::vector<Eigen::VectorXi>& data, 
               size_t epochs = 100, 
               size_t k = 1,
               double lr = 0.1) {
        
        std::cout << "Epoch\tError\tFree Energy" << std::endl;
        
        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            for (const Eigen::VectorXi& v0 : data) {
                const Eigen::VectorXi& h0 = this->sample_hidden(v0);
                const Eigen::MatrixXi& pos_grad = v0 * h0.transpose(); 

                const Eigen::VectorXi& vk = this->CD(v0, k); 
                const Eigen::VectorXi& hk = this->sample_hidden(vk);
                const Eigen::MatrixXi& neg_grad = vk * hk.transpose(); 

                this->weights += lr * (pos_grad - neg_grad).cast<double>();
                this->v_bias += lr * (v0 - vk).cast<double>();
                this->h_bias += lr * (h0 - hk).cast<double>();
            }

            if (epoch % (epochs / 100) == 0) {
                double error = this->reconstruction_error(data);
                double energy = this->avg_free_energy(data);
                std::cout << epoch << "\t" << error << "\t" << energy << std::endl;
            }
        }
    }


    double reconstruction_error(const std::vector<Eigen::VectorXi>& data) {
        double total_error = 0.0;
        for (const auto& sample : data) {
            auto reconstructed = this->CD(sample, 1);
            total_error += (sample - reconstructed).squaredNorm();
        }
        return total_error / data.size();
    }

    double free_energy(const Eigen::VectorXi& v) {
        const auto& hidden_acts = this->hidden_act(v); 
        const auto& log_probs = hidden_acts.unaryExpr(
            [](double act) { return std::log1p(std::exp(act)); }
        );
        return - v.cast<double>().dot(this->v_bias) - log_probs.sum();
    }

    double avg_free_energy(const std::vector<Eigen::VectorXi>& data) {
        double total = 0.0;
        for (const auto& sample : data) total += free_energy(sample);
        return total / data.size();
    }


    void visualize_weights() {       
        for (size_t h = 0; h < this->n_hidden; ++h) {
            for (size_t v = 0; v < this->n_visible; ++v) {
                double weight = this->weights(v, h);
                char c = ' ';
                if (weight > 0.5) c = '#';
                else if (weight > 0.0) c = '.';
                else if (weight < -0.5) c = '-';
                else if (weight < -0.0) c = '~';
                std::cout << c << " ";
            }
            std::cout << std::endl; 
        }
    }

private: 

    void init_params() {
        this->v_bias = Eigen::VectorXd::Zero(this->n_visible);
        this->h_bias = Eigen::VectorXd::Zero(this->n_hidden);

        this->weights = Eigen::MatrixXd::Zero(this->n_visible, this->n_hidden);
        double stddev = std::sqrt(2.0 / (this->n_visible + this->n_hidden));
        std::normal_distribution<double> dist(0.0, stddev);        
        for (size_t i = 0; i < this->n_visible; ++i) {
            for (size_t j = 0; j < this->n_hidden; ++j) {
                this->weights(i, j) = dist(this->rng);
            }
        }
    }


    inline Eigen::VectorXd visible_act(const Eigen::VectorXi& h) const {
        return this->weights * h.cast<double>() + this->v_bias; 
    }

    inline Eigen::VectorXd hidden_act(const Eigen::VectorXi& v) const {
        return this->weights.transpose() * v.cast<double>() + this->h_bias;
    }


    inline Eigen::VectorXd visible_prob(const Eigen::VectorXi& h) const {
        return sigmoid(this->visible_act(h)); 
    }

    inline Eigen::VectorXd hidden_prob(const Eigen::VectorXi& v) const {
        return sigmoid(this->hidden_act(v)); 
    }


    Eigen::VectorXi sample_visible(const Eigen::VectorXi& h) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const auto& randoms = Eigen::VectorXd::NullaryExpr(
            n_visible, [&dist, this]() { return dist(this->rng); }
        );
        return (this->visible_prob(h).array() > randoms.array()).cast<int>();
    }

    Eigen::VectorXi sample_hidden(const Eigen::VectorXi& v) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const auto& randoms = Eigen::VectorXd::NullaryExpr(
            n_hidden, [&dist, this]() { return dist(this->rng); }
        );
        return (this->hidden_prob(v).array() > randoms.array()).cast<int>();
    }

    inline Eigen::VectorXi gibbs_sample(const Eigen::VectorXi& v) {
        return this->sample_visible(this->sample_hidden(v));
    }

    Eigen::VectorXi CD(const Eigen::VectorXi& v, size_t k) {
        auto vk = v;
        for (size_t i = 0; i < k; ++i) vk = this->gibbs_sample(vk);
        return vk; 
    }

}; 
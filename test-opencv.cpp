#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <complex>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

cv::Mat img_prep(const cv::Mat& image) {
    cv::Mat pixel_values(image.rows, image.cols, CV_32FC2);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            float value = pixel[2] + pixel[1] * 256 + pixel[0] * 256 * 256;
            float imaginary_part = 0.0;
            pixel_values.at<cv::Vec2f>(i, j) = cv::Vec2f(value, imaginary_part);
        }
    }
    CV_Assert(pixel_values.type() == CV_32FC2);
    return pixel_values;
}

void show_2_images(const cv::Mat& data1, const cv::Mat& data2, const std::string& title1 = "Modulo", const std::string& title2 = "Phase") {
    if (data1.size() == data2.size() && data1.type() == data2.type()) {
        cv::namedWindow(title1, cv::WINDOW_NORMAL);
        cv::namedWindow(title2, cv::WINDOW_NORMAL);

        cv::imshow(title1, data1);
        cv::imshow(title2, data2);

        cv::waitKey(0);
        cv::destroyAllWindows();
    } else {
        std::cerr << "Изображения должны быть одинакового размера и типа." << std::endl;
    }
}

//Random complex field generation
cv::Mat generate_random_complex_field(int width = 555, int height = 555) {
    cv::Mat rand_field(height, width, CV_32FC2);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> amplitude_dist(0, (1 << 24) - 1);
    std::uniform_real_distribution<float> phase_dist(0.0, 2.0 * M_PI);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int amplitude_int = amplitude_dist(gen);
            float amplitude = static_cast<float>(amplitude_int);
            float phase = phase_dist(gen);
            float real_part = amplitude * std::cos(phase);
            float imaginary_part = amplitude * std::sin(phase);
            rand_field.at<cv::Vec2f>(i, j) = cv::Vec2f(real_part, imaginary_part);
        }
    }
    CV_Assert(rand_field.type() == CV_32FC2);
    return rand_field;
}

cv::Mat FT(const cv::Mat& data) {
    CV_Assert(data.type() == CV_32FC2);
    cv::Mat ftmatrix;
    cv::dft(data, ftmatrix);
    return ftmatrix;
}

cv::Mat IFT(const cv::Mat& data) {
    CV_Assert(data.type() == CV_32FC2);
    cv::Mat iftmatrix;
    cv::idft(data, iftmatrix, cv::DFT_SCALE);
    return iftmatrix;
}

cv::Mat steps(const cv::Mat& X_inp, const cv::Mat& X_source) {
    CV_Assert(X_inp.type() == CV_32FC2 && X_source.type() == CV_32FC2);
    cv::Mat ft_X_inp = FT(X_inp);
    cv::Mat planes[2];
    cv::split(ft_X_inp, planes); // planes[0] — Re, planes[1] — Im
    cv::Mat amplitude, phase;
    cv::cartToPolar(planes[0], planes[1], amplitude, phase);
    cv::Mat source_planes[2];
    cv::split(X_source, source_planes); // source_planes[0] — Re, source_planes[1] — Im
    cv::Mat new_amplitude;
    cv::magnitude(source_planes[0], source_planes[1], new_amplitude);
    cv::Mat new_ft_X_inp;
    cv::Mat real_part, imag_part;
    cv::polarToCart(new_amplitude, phase, real_part, imag_part);
    cv::merge(std::vector<cv::Mat>{real_part, imag_part}, new_ft_X_inp);
    cv::Mat result = IFT(new_ft_X_inp);
    return result;
}

cv::Mat applyMaskToComplex(const cv::Mat& complexData, const cv::Mat& mask) {
    cv::Mat planes[2];
    cv::split(complexData, planes); // planes[0] — Re, planes[1] — Im
    cv::Mat amplitude, phase;
    cv::cartToPolar(planes[0], planes[1], amplitude, phase);
    cv::Mat new_amplitude = amplitude.mul(mask);
    cv::Mat new_phase = phase.mul(mask);
    cv::Mat real_part, imag_part;
    real_part.create(new_amplitude.size(), CV_32F);
    imag_part.create(new_phase.size(), CV_32F); 
    cv::polarToCart(new_amplitude, new_phase, real_part, imag_part);
    cv::Mat result;
    cv::merge(std::vector<cv::Mat>{real_part, imag_part}, result);

    return result;
}

std::pair<cv::Mat, float> ER(int N_iter, const cv::Mat& target, const cv::Mat& source, const cv::Mat& mask) {
    cv::Mat A = target.clone();
    cv::Mat D;
    float D_norm, A_norm, Error;

    for (int i = 0; i < N_iter; ++i) {
        D = steps(A, source); 
        A = applyMaskToComplex(D, mask); 
        D_norm = cv::norm(D);
        A_norm = cv::norm(A);
        Error = std::sqrt(D_norm * D_norm - A_norm * A_norm) / D_norm;
    }
    return { A, Error };
}

cv::Mat HIO(int N_iter, float beta, const cv::Mat& Target, const cv::Mat& Source, const cv::Mat& mask, const cv::Mat& antimask) {
    cv::Mat A = Target.clone();
    cv::Mat D;

    for (int i = 0; i < N_iter; ++i) {
        D = steps(A, Source);
//        cv::Mat masked_D = applyMaskToComplex(D, mask);
//        cv::Mat masked_A = applyMaskToComplex(A - beta * D, antimask);
        A = applyMaskToComplex(D, mask) + applyMaskToComplex(A - beta * D, antimask);
    }
    return A;
}
std::pair<cv::Mat, float> retrieving(const cv::Mat& img, const cv::Mat& real_img, float beta, int N, const cv::Mat& mask, const cv::Mat& antimask) {
    cv::Mat c_hio = HIO(30, beta, real_img, img, mask, antimask);
    std::pair<cv::Mat, float> er_result = ER(N, c_hio, img, mask);
    return er_result;
}

// retr_block
std::pair<cv::Mat, float> retr_block(const cv::Mat& inp, int N, const cv::Mat& crypt_values, const cv::Mat& mask, const cv::Mat& antimask) {
    std::pair<cv::Mat, float> result1 = retrieving(crypt_values, inp, 1.0f, N, mask, antimask);
    std::pair<cv::Mat, float> result2 = retrieving(crypt_values, result1.first, 0.7f, N, mask, antimask);
    std::pair<cv::Mat, float> result3 = retrieving(crypt_values, result2.first, 0.4f, N, mask, antimask);
    std::pair<cv::Mat, float> result4 = retrieving(crypt_values, result3.first, 0.1f, N, mask, antimask);
    return result4;
}

int main() {
    std::string img_path = "crypt.jpg";
    cv::Mat crypt = cv::imread(img_path, cv::IMREAD_COLOR);

    if (crypt.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    cv::Mat crypt_values = img_prep(crypt);
    int size = 555;
    int center = size / 2;
    int half_size = size / 4;

    // mask
    cv::Mat strict_mask = cv::Mat::zeros(size, size, CV_32F);
    int center_square_start = center - half_size;
    int center_square_end = center + half_size;

    // Заполняем центральную область маски единицами
    strict_mask(cv::Range(center_square_start, center_square_end + 1),
        cv::Range(center_square_start, center_square_end + 1)) = 1.0;

    // Создаем антимаску
    cv::Mat antimask = 1.0 - strict_mask;

    float error = 1.0f;
    cv::Mat field_1;
    while (error > 0.078f) {
        cv::Mat random_field = generate_random_complex_field();

        std::pair<cv::Mat, float> result = retr_block(random_field, 10, crypt_values, strict_mask, antimask);
        field_1 = result.first;
        error = result.second;

        for (int i = 0; i < 8; ++i) {
            cv::Mat abs_random_field;
            cv::magnitude(random_field, cv::Mat::zeros(random_field.size(), CV_32FC2), abs_random_field);
            std::pair<cv::Mat, float> new_result = retr_block(abs_random_field, 10, crypt_values, strict_mask, antimask);
            field_1 = new_result.first;
            error = new_result.second;
        }

        cv::Mat abs_random_field;
        cv::magnitude(random_field, cv::Mat::zeros(random_field.size(), CV_32FC2), abs_random_field);
        std::pair<cv::Mat, float> final_result = retr_block(abs_random_field, 200, crypt_values, strict_mask, antimask);
        field_1 = final_result.first;
        error = final_result.second;

        std::cout << "Error: " << error << std::endl;
    }

    
    cv::Mat abs_field, phase_field;
    cv::Mat planes[2];
    cv::split(field_1, planes);
    cv::magnitude(planes[0], planes[1], abs_field); 
    cv::phase(planes[0], planes[1], phase_field);  
    cv::normalize(abs_field, abs_field, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(phase_field, phase_field, 0, 255, cv::NORM_MINMAX, CV_8U);
    show_2_images(abs_field, phase_field, "Modulo", "Phase");

    return 0;
}


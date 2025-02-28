
#ifndef R250_H
#define R250_H

#ifdef __cplusplus
extern "C" {
#endif

	void r250_init(int seed);
	unsigned int r250();
	unsigned int r250n(unsigned n);
	double dr250();

#ifdef __cplusplus
}
#endif

#endif // R250_H
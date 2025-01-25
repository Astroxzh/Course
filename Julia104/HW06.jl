### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 250252f0-6e61-11ec-2ffe-ebddaa9f0fb3
begin
	using Pkg 
	# Pkg.activate("../../")
	using Revise
end

# ╔═╡ 0eeb5bb1-bb41-43cc-8c14-eb9c1ebf6fad
using FourierTools, IndexFunArrays, ImageShow, FFTW, TestImages, Colors, ColorTypes

# ╔═╡ 1b2ebe1a-f3a7-494c-bcca-18df5403ee65
using PlutoTest, Noise

# ╔═╡ 8ce61783-2933-44b9-8660-0d7925b6c0cc
"""
    complex_show(arr)

Displays a complex array. Color encodes phase, brightness encodes magnitude.
Works within Jupyter and Pluto.
"""
function complex_show(cpx::AbstractArray{<:Complex, N}) where N
	ac = abs.(cpx)
	HSV.(angle.(cpx)./2pi*256,ones(Float32,size(cpx)),ac./maximum(ac))
end

# ╔═╡ 23975df0-b357-46c2-8935-4c8111b7c197
"""
    gray_show(arr; set_one=false, set_zero=false)
Displays a real gray color array. Brightness encodes magnitude.
Works within Jupyter and Pluto.

## Keyword args
* `set_one=false` divides by the maximum to set maximum to 1
* `set_zero=false` subtracts the minimum to set minimum to 1
"""
function gray_show(arr; set_one=true, set_zero=false)
    arr = set_zero ? arr .- minimum(arr) : arr
    arr = set_one ? arr ./ maximum(arr) : arr
    Gray.(arr)
end

# ╔═╡ 4b46e310-cc26-4a96-8c2e-0c89307d4e34
md"# Structured Illumination Microscopy

In this homework we are going to work through the full pipeline from capturing a dataset of a SIM to reconstructing the data.

For the demonstration we focus on the 2D case and consider the resolution enhancement only along one direction. That allows us to observe the effect clearly.
"

# ╔═╡ 7054db2c-606e-48eb-ab93-8c0260cb7a81
md"## 1 Simulate PSF"

# ╔═╡ edaacc75-fd91-4d46-a31f-21e738253708
"""
	simulate_PSF(s, r, [T=Float32])

Simulate the incoherent 2D PSF with radius `r` and total output size `s`.
This functions returns the PSF centered around the first entry (top left corner).
Furthermore, it returns the `freq_limit` of the PSF.
The `freq_limit` is the frequency at which the OTF becomes 0.

The procedure is as following:
* create a disc with values of 1 and radius `r` (use `rr2`)
* take the `ift` of it.
* abs2.
* normalize the sum to 1
* center pixcel should be located in the top left
"""
function simulate_PSF(s, r, T=Float32)
	# TODO calculate real PSF
	# psf = zeros(T, s)
	circ = rr2(s) .<= r^2
	psf = abs2.(ift(circ))
	psf ./= sum(psf)
	psf = ifftshift(psf)
	
	# don't change freq_limit, is already correct
	freq_limit = r / (s[1] / 2)
	return psf, freq_limit
end

# ╔═╡ cd5bef41-6246-4d88-91d9-c83b7a47e110
# leave this value equal to 40!
radius = 40

# ╔═╡ 29e6311e-eb5b-4073-81dd-7372ac06498e
psf, freq_limit = simulate_PSF((512,512), radius);

# ╔═╡ c4c8b57b-e09a-43f8-b371-bb1f0761215d
gray_show(psf)

# ╔═╡ 599aabf7-6333-4272-ae07-fec07bdb02aa
img = Float32.(testimage("resolution_test_512"));

# ╔═╡ 920c26f4-5750-4c83-871d-9622b3dca134
gray_show(img);

# ╔═╡ a550fc7f-ff7c-4fd5-9e40-d8ed00da200f
md"### 1 Test"

# ╔═╡ 7cba5a99-21a7-45d6-a92a-7eb1e75e39d3
PlutoTest.@test simulate_PSF((5, 5), 2)[1] ≈ (Float32[0.36 0.10472136 0.01527864 0.01527864 0.10472136; 0.10472136 0.030462677 0.0044444446 0.0044444446 0.030462677; 0.01527864 0.0044444446 0.00064843567 0.00064843567 0.0044444446; 0.01527864 0.0044444446 0.00064843567 0.00064843567 0.0044444446; 0.10472136 0.030462677 0.0044444446 0.0044444446 0.030462677], 0.8)[1]

# ╔═╡ eafd2e8a-6bbd-4fe4-a5fd-d721b78604e1
PlutoTest.@test simulate_PSF((5, 5), 2)[2] == 0.8

# ╔═╡ 75f2ea94-ba30-48d3-b676-8423f14c453c
delta = zeros((5,5));

# ╔═╡ 9b856e9c-b818-4e49-a27d-0c0db623698e
delta[1] = 1;

# ╔═╡ 11009f9c-de4c-4ade-ad30-375ed47c3901
PlutoTest.@test simulate_PSF((5, 5), 3)[1] ≈ delta

# ╔═╡ 4ff25625-e450-47e4-97ae-bb24b903ed98
md"## 2 Structured Illumination

The key idea behind SIM is to illuminate the sample with an illumination pattern of the form

$I_n = 1 + \cos(k_g \cdot x + \theta_n)$

where $k_g$ is the frequency of the illumination pattern (twice the grating constant). $\theta$ is a illumination pattern shift which is different for each image. Such an illumination allows to increase the resolution along the first dimension.

We capture three images ($n=\{1,2,3\}$) of the sample which can be expressed as

$Y_n = (I_n \cdot S) * h$
where $h$ is the PSF, $S$ the ideal sample and $Y$ the measurement.

In our example we choose the following 3 different illuminations

$I_{1} = 1 + \cos\left(k_g \cdot x \cdot 2 \pi + 1\cdot \frac{2 \pi}{3} \right)$

$I_2 = 1 + \cos\left(k_g \cdot x  \cdot 2 \pi+ 2 \cdot \frac{2 \pi}{3}\right)$

$I_3 = 1 + \cos\left(k_g \cdot x  \cdot 2 \pi+ 3 \cdot \frac{2 \pi}{3}\right)$


"

# ╔═╡ ca8618e8-8389-4d30-8112-af0e987de417
"""
	illuminate(arr, kg)

Returns a vector of `arr` illuminated three times according to the SIM principle.
Implement the equations from above. Only illumination, no convolution with PSF.

You should return a vector of the three illumination arrays.
`[I_1, I_2, I_3]`.
"""
function illuminate(arr, kg)
	# x coordinates
	x = 1:size(arr, 1)
	# todo fix illumination
	I₁ = 1.0 .+ cos.(kg .* x .* 2π .+ 1 * 2π/3)
	I₂ = 1.0 .+ cos.(kg .* x .* 2π .+ 2 * 2π/3)
	I₃ = 1.0 .+ cos.(kg .* x .* 2π .+ 3 * 2π/3)

	I_1 = arr .* I₁
	I_2 = arr .* I₂
	I_3 = arr .* I₃

	# todo return correct results
	return[I_1, I_2, I_3]
end

# ╔═╡ 20aaf9b2-2660-413f-90a8-1f4fdc91d8c7
# toy example
illuminate(ones((4,4)), 0.5)

# ╔═╡ 4898da63-4da0-4538-b08d-cd4fe419584b
Illus = illuminate(ones((512, 512)), freq_limit * 0.8);

# ╔═╡ 8457cfeb-705f-41e8-a203-298c8ee4a7ab
# display the three illuminations
gray_show(hcat(Illus...))

# ╔═╡ 28784e50-285f-47a8-a311-4122ec8f47ce
# check whether illumination still pass through the system.
# if you see the same stripes but with lower contrast, then everything
# is perfect
gray_show(conv(Illus[1], psf))

# ╔═╡ 6e2814c9-5644-47b6-8a6e-3308e1623d5e
md"### 2 Test"

# ╔═╡ ca0e55ca-ec4f-4cd8-a1f7-a2a63ba9782d
PlutoTest.@test illuminate(ones((4, 4)), 0.5) ≈ [[1.4999999999999993 1.4999999999999993 1.4999999999999993 1.4999999999999993; 0.5000000000000008 0.5000000000000008 0.5000000000000008 0.5000000000000008; 1.4999999999999991 1.4999999999999991 1.4999999999999991 1.4999999999999991; 0.500000000000001 0.500000000000001 0.500000000000001 0.500000000000001], [1.5000000000000004 1.5000000000000004 1.5000000000000004 1.5000000000000004; 0.49999999999999867 0.49999999999999867 0.49999999999999867 0.49999999999999867; 1.5000000000000013 1.5000000000000013 1.5000000000000013 1.5000000000000013; 0.49999999999999845 0.49999999999999845 0.49999999999999845 0.49999999999999845], [0.0 0.0 0.0 0.0; 2.0 2.0 2.0 2.0; 0.0 0.0 0.0 0.0; 2.0 2.0 2.0 2.0]]

# ╔═╡ fa957648-0b4a-48eb-96d5-bb731966693c
Ittt = illuminate(ones((10, 10)), 1/3);

# ╔═╡ 287d85ae-efba-4095-b7ce-58d90bd22492
PlutoTest.@test Ittt[1] .+ Ittt[2] .+ Ittt[3] ≈ 3 .* ones((10, 10))

# ╔═╡ fce697cc-67d1-4373-b2b0-a6a60b8a5893
PlutoTest.@test all(illuminate([1 1; 0 0], 0) .≈ [[0.49999999999999956 0.49999999999999956; 0.0 0.0], [0.5000000000000002 0.5000000000000002; 0.0 0.0],  [2.0 2.0; 0.0 0.0],])

# ╔═╡ b961b3e0-6935-4f29-9f94-be6e21f302e3
PlutoTest.@test  illuminate([1 1; 0 0], 0.2) ≈ [[0.02185239926619431 0.02185239926619431; 0.0 0.0], [1.6691306063588578 1.6691306063588578; 0.0 0.0], [1.3090169943749477 1.3090169943749477; 0.0 0.0]]


# ╔═╡ 6e904b1d-2071-4d48-ad21-ab180e333474
md"## 3 Forward Imaging Process
Now we complete the forward modelling part from the image to the three measurements.

The grating constant should be small enough such that illumination 
passes through the system.
Since the optical system has a bandlimit, not abritrary small illumination patterns pass through the system.
The larger you choose it, the more of the reconstruction is affected by noise.
By  selecting `0.8 * freq_limit` we are definitely below the frequency threshold (e.g. 1.1 would be above)
"

# ╔═╡ e3994f91-a7bc-4811-b6e6-1f67692ff554
# don't change!
kg = 0.8 * freq_limit

# ╔═╡ 71af2b63-475c-429f-a15e-9a212f724a69
# the amount the three OTFs are displaced with respect to each other in Fourier space
# we need the value later for the reconstruction
# this value should stay at 64!
fourier_space_shift = radius * 2 * 0.8

# ╔═╡ 2a78c81d-f8f9-4d99-9c24-1aefb9941294
"""
	forward(img, psf, kg)

From the input `img` create three output images.
The output images are produced by multiplying the three different illumination
to the input image and convolving the three resulting `img`s with the `psf`.
Apply `poisson` to each of the image with a photon number of 1000 (use `N_phot`).
Use the function `poisson` for that.

## Procedure
* Multiplication with illumination
* convolution (use `FourierTools.conv`) with `psf`.
* Add noise with `poisson(img_without_noise, N_phot)`
* Return vectors of the three measuremens,e g. `[meas_1, meas_2, meas_3]`

"""
function forward(img, psf, kg, N_phot=1000)
	# todo illuminate img and convolve with PSF
	img_illu = illuminate(img, kg)
	img_without_noise1 = FourierTools.conv(img_illu[1], psf)
	img_without_noise2 = FourierTools.conv(img_illu[2], psf)
	img_without_noise3 = FourierTools.conv(img_illu[3], psf)
	
	meas_1 = poisson(img_without_noise1, N_phot)
	meas_2 = poisson(img_without_noise2, N_phot)
	meas_3 = poisson(img_without_noise3, N_phot)
	# return correct result
	return [meas_1, meas_2, meas_3]
end

# ╔═╡ d68d9f24-bcea-4bcc-b597-3c051e77d99c
Is = forward(img, psf, kg);

# ╔═╡ 23c98291-4c72-4a60-b722-910eadaf7e77
md"Three images which look all slightly different because"

# ╔═╡ 9a75e192-9605-4787-acee-706ee7c90549
gray_show(hcat(Is...))

# ╔═╡ 524b208d-7fc3-4d96-bdc1-2bed72f2a9ee
md"In comparison without any structured illumination"

# ╔═╡ 2c1a5b9e-8b1c-4619-b303-69d29181a5c4
[gray_show(conv(img, psf)) gray_show(Is[1])]

# ╔═╡ 09593275-85bf-42ee-860e-ebbb74866a40
md"## 3 Test"

# ╔═╡ 1103b35d-cb86-4970-beb7-c8463e374f78
# test random example
PlutoTest.@test all(.≈(forward([1 2; 3 4], [0.55 0.15; 0.15 0.15], 0.1, 100_000_000),  [[0.09645833671795113 0.13103484317669054; 0.08808351198799098 0.09683198111576993], [2.6912285614539946 3.133633214436668; 4.25256575017253 4.919802755223418], [2.9120113512125654 3.6367083598655254; 3.7589874298841632 4.283291164312841]], rtol=0.01))

# ╔═╡ ba74ec4a-b38f-4fcb-9b43-2448976153eb
PlutoTest.@test radius == 40

# ╔═╡ ab56310f-671d-4b54-b9c2-9be102c826e3
PlutoTest.@test kg == 1/8

# ╔═╡ db20d906-b22f-4f16-b355-eba9c82809d7
PlutoTest.@test fourier_space_shift == 64

# ╔═╡ b7f11d85-6859-4b5a-814e-003946fbd894
md"# 4 Unmixing
The three measurements are not the different parts of the Fourier spectrum yet.
But they contain the information to high frequencies!
Therefore, we want to unmix the three parts of the Fourier space.
"

# ╔═╡ 96b8f1e6-9159-4616-be08-c88f17e9829e
"""
	extract_components(Is)

Extract the different Fourier components from the 3 measurements (the vector `Is`).
This is achieved by inverting the mixing matrix (`inv(M)`) and multiplying (use a `*` for matrix multiplication, not a pointwise `.*`) it from the left to
`[fft(Is[1]), fft(Is[2]), fft(Is[3])]`.

See the lecture slides for more information.
"""
function extract_components(Is)
	# TODO add correct angles
	θ = [2π/3, 2*2π/3, 3*2π/3]
	
	# TODO create correct matrix
	M = [exp(-1im * θ[1]) 1 exp(1im * θ[1]);
		 exp(-1im * θ[2]) 1 exp(1im * θ[2]);
		 exp(-1im * θ[3]) 1 exp(1im * θ[3])]

	IsC = [fft(Is[i]) for i in 1:3]

	unmixing = inv(M) * IsC

	# TODO return correct result
	return unmixing
end

# ╔═╡ 53af3f77-cbc2-4809-8dda-743e096df10c
Cₙ = extract_components(Is);

# ╔═╡ 4c136f35-e489-4d6e-83e0-e153bfc7040d
md"Here you can see the three different components still. They are not shifted to the correct position, yet.
"

# ╔═╡ bf97497a-e149-4552-9bdd-9fbeb5c24987
gray_show(log1p.(abs.(hcat(fftshift.(Cₙ)...))))

# ╔═╡ c922f283-3b31-43bd-9855-933be8a9e095
md"### 4 Test"

# ╔═╡ 460a8ea2-d077-48fc-804e-7367ac16eff8
# check if three output array
PlutoTest.@test size(extract_components([randn((2,2)) for i = 1:3])) == (3,)

# ╔═╡ ba0e90f8-a310-40f9-a306-fb8700ea473f
# check if structure and type is ok
PlutoTest.@test typeof(extract_components([randn((2,2)) for i = 1:3])) == Vector{Matrix{ComplexF64}}

# ╔═╡ 5d263158-03b0-4c87-ad90-a81c24493b66
# check a random example for correctness
PlutoTest.@test extract_components([[1 2; 3 4], [1 3; 4 5], [1 10; 20 30]]) ≈ Matrix{ComplexF64}[[16.50000000000001 - 0.8660254037844412im -5.500000000000003 + 0.2886751345948141im; -11.500000000000005 + 0.28867513459481564im 0.5000000000000002 + 0.28867513459481253im], [28.00000000000001 + 1.5597088666161704e-15im -8.000000000000002 - 3.563506943082403e-16im; -16.000000000000007 - 6.01679086153965e-16im 1.1102230246251565e-16 - 1.1102230246251565e-16im], [16.5 + 0.8660254037844446im -5.5 - 0.2886751345948149im; -11.5 - 0.28867513459481736im 0.5 - 0.2886751345948127im]]

# ╔═╡ 515d2174-56e3-4ebc-a475-b84f911a5a2f
md"# 5 Reconstruction
Having the three parts of the Fourier space separated, we can try to combine them in a reasonable way to one Fourier spectrum.
Simply adding them would not be optimal with respect to Signal-to-Noise-Ratio (SNR).
"

# ╔═╡ a97d7c94-e69b-44cd-ae4c-61d8d86808a5
"""
	reconstruct(psf, Cₙ, fourier_space_shift)

`psf` is the PSF of the single image (without SIM part).
`Cₙ` is an vector containing the three components of the Fourier spectrum.
`fourier_space_shift` is the shift the three pictures are shifted with respect to the center.

## Procedure
* First shift the spectrum (`otf = fft(psf)`) by `fourier_space_shift` along the first dimension (`circshift`)
* apply the same shift for the weighting factors
* Combine the Fourier spectra using _weighted averaging_ (see slides)
* also return the effective OTF
"""
function reconstruct(psf, Cₙ, fourier_space_shift)
	# Int shift
	Δ = round(Int, fourier_space_shift)

	otf = fft(psf)

	otf₋₁ = circshift(otf, (Δ, 0))
	otf₀ = otf
	otf₁ = circshift(otf, (-Δ, 0))
	
	# TODO fix the weights
	w₋₁ = 0.5 * otf₋₁
	w₀ = otf
	w₁ = 0.5 * otf₁

	# TODO fix the mixing 
	C₋₁ = Cₙ[1]
	C₀ = Cₙ[2]
	C₁ = Cₙ[3]
	
	# the small 1f-8 factor is added to prevent division by zero
	res_fourier_space = (w₋₁ .* C₋₁ .+ w₀ .* C₀ .+ w₁ .* C₁ .+ 1f-8^2) ./ 
		  (w₋₁ .+ w₀ .+ w₁ .+ 1f-8)

	# todo calculate similarly to res the eff_otf but adapt it according to the slides
	eff_otf = (w₋₁ .* otf₋₁ .+ w₀ .* otf₀ .+ w₁ .* otf₁) ./ 
              (w₋₁ .+ w₀ .+ w₁ .+ 1f-8)


	# todo adapt res_fourier_space to be a real space image
	res = real(ifft(res_fourier_space))

	# don't change return args
	return res, eff_otf
end

# ╔═╡ 9e64def4-b397-4292-8fe9-8a828e08ee0a
res, eff_otf = reconstruct(psf, Cₙ, fourier_space_shift);

# ╔═╡ 54fa3a01-5a8d-4dd3-a521-a60a2f69b541
md"Unfiltered reconstruction"

# ╔═╡ d9c33b8c-6ccf-462f-b5c2-267fd7b93ba3
gray_show(res)

# ╔═╡ 2f14fb93-91bc-44af-8a6d-cebaa1607a51
md"Unfiltered Fourier spectrum"

# ╔═╡ fd53c66d-ba90-4d96-9a11-eabc1dd666dc
gray_show(log1p.(abs.(ft(res))))

# ╔═╡ 19546bad-d7be-4c09-a248-f7e98c53d42d
"""
	wiener_filter(img, otf, ϵ)

Copy the wiener filter from previous homeworks and adapt it (OTF instead of the PSF is passed).

"""
function wiener_filter(img, otf, ϵ)
	# todo
	# fix wiener filter but use otf instead of PSF (look old homework solutions)
	imgFre = fft(img)

    wienerFilterFre = conj(otf) ./ (abs2.(otf) .+ ϵ)  # Wiener filter formula

    restoredFre = imgFre .* wienerFilterFre

    restoredImg = real(ifft(restoredFre))

	return restoredImg
end

# ╔═╡ 33c156b6-5c18-4377-9920-32c7865898e9
md"### Final Inspection"

# ╔═╡ 0cdf07a5-a87e-449e-ab09-871924cee6d1
md"
If everything is correct, we should definitely see an resolution improvement in the left image (the SIM reconstruction).
The resolution increase is mainly along the first dimension.
"

# ╔═╡ 0a729667-599a-41f1-98f3-ff94608c15bb
[ gray_show(wiener_filter(res, eff_otf, 1e-3)) gray_show(wiener_filter(poisson(conv(img, psf), 300), fft(psf), 1e-3)) gray_show(img)]

# ╔═╡ 0d78c765-1912-4ded-a5e8-aa98afe73cb4
gray_show(log1p.(abs.(ft(wiener_filter(res, eff_otf, 1e-1)))));

# ╔═╡ 5566b0c6-ac96-4342-a62b-0c292469c2eb
[gray_show(wiener_filter(res, eff_otf, 1e-3)) gray_show(wiener_filter(poisson(conv(img, psf), 300), fft(psf), 1e-3))]

# ╔═╡ 3bed0c72-ab5c-4dbf-85a5-6f6d062dec97
md"The left image shows the effective OTF for the SIM system.
On the right we can see the OTF of the same microscope without structured illumination"

# ╔═╡ d32536af-b369-4b39-87f6-4e1172d971ae
[gray_show(fftshift(log1p.(abs.(eff_otf))))  gray_show(log1p.(abs.(ffts(psf))))]

# ╔═╡ 35cdd4f9-87f7-408e-a3fd-7456b7238dd5
md"## 5 Test"

# ╔═╡ 1320b506-d378-4fa1-bbf1-c2f994abfff1
PlutoTest.@test wiener_filter([1, 2, 3, 4, 5], [1, -2, 3, 4, 0], 0.001) ≈ [2.9553001744303087, 3.5816540338359086, 2.9970029970029977, 2.4123519601700862, 3.0387058195756858]

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FourierTools = "b18b359b-aebc-45ac-a139-9c0ccbb2871e"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
IndexFunArrays = "613c443e-d742-454e-bfc6-1d7f8dd76566"
Noise = "81d43f40-5267-43b7-ae1c-8b967f377efa"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
ColorTypes = "~0.11.5"
Colors = "~0.13.0"
FFTW = "~1.8.0"
FourierTools = "~0.4.6"
ImageShow = "~0.3.8"
IndexFunArrays = "~0.2.7"
Noise = "~0.3.3"
PlutoTest = "~0.2.2"
Revise = "~3.6.4"
TestImages = "~1.9.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "9702ed5b005743f5bd54985e81bfe9360049f652"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractNFFTs]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "292e21e99dedb8621c15f185b8fdb4260bb3c429"
uuid = "7f219486-4aa7-41d6-80a7-e08ef20ceed7"
version = "0.8.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "96bed9b1b57cf750cca50c311a197e306816a1cc"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.39"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BasicInterpolators]]
deps = ["LinearAlgebra", "Memoize", "Random"]
git-tree-sha1 = "3f7be532673fc4a22825e7884e9e0e876236b12a"
uuid = "26cce99e-4866-4b6d-ab74-862489e035e0"
version = "0.7.1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FourierTools]]
deps = ["ChainRulesCore", "FFTW", "IndexFunArrays", "LinearAlgebra", "NDTools", "NFFT", "PaddedViews", "Reexport", "ShiftedArrays"]
git-tree-sha1 = "146f9bff9647a279e6a5053d1a48c04e67051d1a"
uuid = "b18b359b-aebc-45ac-a139-9c0ccbb2871e"
version = "0.4.6"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0224cce99284d997f6880a42ef715a37c99338d1"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.2+2"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "c5c5478ae8d944c63d6de961b19e6d3324812c35"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.4.0"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fa01c98985be12e5d75301c4527fff2c46fa3e0e"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "7.1.1+1"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndexFunArrays]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f78703c7a4ba06299cddd8694799c91de0157ac"
uuid = "613c443e-d742-454e-bfc6-1d7f8dd76566"
version = "0.2.7"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef10afc9f4b942bcd75f4c3bc9d9e8d802944c23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.0+2"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "10da5154188682e5c0726823c2b5125957ec3778"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.38"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a7f43994b47130e4f491c3b2dbe78fe9e2aed2b3"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.0+2"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NDTools]]
deps = ["LinearAlgebra", "OffsetArrays", "PaddedViews", "Random", "Statistics"]
git-tree-sha1 = "3e5105ea7d08354014613c96bdfeaa0d151f1c1a"
uuid = "98581153-e998-4eef-8d0d-5ec2c052313d"
version = "0.7.1"

[[deps.NFFT]]
deps = ["AbstractNFFTs", "BasicInterpolators", "Distributed", "FFTW", "FLoops", "LinearAlgebra", "PrecompileTools", "Printf", "Random", "Reexport", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "d28544d20956835b9d856ff240aa61f898a00652"
uuid = "efe261a4-0d2b-5849-be55-fc731d526b0d"
version = "0.13.5"

    [deps.NFFT.extensions]
    NFFTGPUArraysExt = ["Adapt", "GPUArrays"]

    [deps.NFFT.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Noise]]
deps = ["ImageCore", "PoissonRandom", "Random"]
git-tree-sha1 = "d34a07459e1ebdc6b551ecb28e3c19993f544d91"
uuid = "81d43f40-5267-43b7-ae1c-8b967f377efa"
version = "0.3.3"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

    [deps.OffsetArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+2"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[deps.PoissonRandom]]
deps = ["Random"]
git-tree-sha1 = "a0f1159c33f846aa77c3f30ebbc69795e5327152"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "470f48c9c4ea2170fd4d0f8eb5118327aada22f5"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.4"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "52af86e35dd1b177d051b12681e1c581f53c281b"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "5b2ca70b099f91e54d98064d5caf5cc9b541ad06"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.3"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "ImageIO", "ImageMagick", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "fc32a2c7972e2829f34cf7ef10bbcb11c9b0a54c"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.9.0"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "3c0faa42f2bd3c6d994b06286bba2328eae34027"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.2"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+3"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+3"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+2"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "7dfa0fd9c783d3d0cc43ea1af53d69ba45c447df"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+3"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═250252f0-6e61-11ec-2ffe-ebddaa9f0fb3
# ╠═0eeb5bb1-bb41-43cc-8c14-eb9c1ebf6fad
# ╠═1b2ebe1a-f3a7-494c-bcca-18df5403ee65
# ╠═8ce61783-2933-44b9-8660-0d7925b6c0cc
# ╠═23975df0-b357-46c2-8935-4c8111b7c197
# ╟─4b46e310-cc26-4a96-8c2e-0c89307d4e34
# ╟─7054db2c-606e-48eb-ab93-8c0260cb7a81
# ╠═edaacc75-fd91-4d46-a31f-21e738253708
# ╠═cd5bef41-6246-4d88-91d9-c83b7a47e110
# ╠═29e6311e-eb5b-4073-81dd-7372ac06498e
# ╠═c4c8b57b-e09a-43f8-b371-bb1f0761215d
# ╠═599aabf7-6333-4272-ae07-fec07bdb02aa
# ╠═920c26f4-5750-4c83-871d-9622b3dca134
# ╟─a550fc7f-ff7c-4fd5-9e40-d8ed00da200f
# ╠═7cba5a99-21a7-45d6-a92a-7eb1e75e39d3
# ╠═eafd2e8a-6bbd-4fe4-a5fd-d721b78604e1
# ╠═75f2ea94-ba30-48d3-b676-8423f14c453c
# ╠═9b856e9c-b818-4e49-a27d-0c0db623698e
# ╠═11009f9c-de4c-4ade-ad30-375ed47c3901
# ╟─4ff25625-e450-47e4-97ae-bb24b903ed98
# ╠═ca8618e8-8389-4d30-8112-af0e987de417
# ╠═20aaf9b2-2660-413f-90a8-1f4fdc91d8c7
# ╠═4898da63-4da0-4538-b08d-cd4fe419584b
# ╠═8457cfeb-705f-41e8-a203-298c8ee4a7ab
# ╠═28784e50-285f-47a8-a311-4122ec8f47ce
# ╟─6e2814c9-5644-47b6-8a6e-3308e1623d5e
# ╠═ca0e55ca-ec4f-4cd8-a1f7-a2a63ba9782d
# ╠═fa957648-0b4a-48eb-96d5-bb731966693c
# ╠═287d85ae-efba-4095-b7ce-58d90bd22492
# ╠═fce697cc-67d1-4373-b2b0-a6a60b8a5893
# ╠═b961b3e0-6935-4f29-9f94-be6e21f302e3
# ╟─6e904b1d-2071-4d48-ad21-ab180e333474
# ╠═e3994f91-a7bc-4811-b6e6-1f67692ff554
# ╠═71af2b63-475c-429f-a15e-9a212f724a69
# ╠═2a78c81d-f8f9-4d99-9c24-1aefb9941294
# ╠═d68d9f24-bcea-4bcc-b597-3c051e77d99c
# ╟─23c98291-4c72-4a60-b722-910eadaf7e77
# ╠═9a75e192-9605-4787-acee-706ee7c90549
# ╟─524b208d-7fc3-4d96-bdc1-2bed72f2a9ee
# ╠═2c1a5b9e-8b1c-4619-b303-69d29181a5c4
# ╟─09593275-85bf-42ee-860e-ebbb74866a40
# ╠═1103b35d-cb86-4970-beb7-c8463e374f78
# ╠═ba74ec4a-b38f-4fcb-9b43-2448976153eb
# ╠═ab56310f-671d-4b54-b9c2-9be102c826e3
# ╠═db20d906-b22f-4f16-b355-eba9c82809d7
# ╟─b7f11d85-6859-4b5a-814e-003946fbd894
# ╠═96b8f1e6-9159-4616-be08-c88f17e9829e
# ╠═53af3f77-cbc2-4809-8dda-743e096df10c
# ╟─4c136f35-e489-4d6e-83e0-e153bfc7040d
# ╠═bf97497a-e149-4552-9bdd-9fbeb5c24987
# ╟─c922f283-3b31-43bd-9855-933be8a9e095
# ╠═460a8ea2-d077-48fc-804e-7367ac16eff8
# ╠═ba0e90f8-a310-40f9-a306-fb8700ea473f
# ╠═5d263158-03b0-4c87-ad90-a81c24493b66
# ╟─515d2174-56e3-4ebc-a475-b84f911a5a2f
# ╠═a97d7c94-e69b-44cd-ae4c-61d8d86808a5
# ╠═9e64def4-b397-4292-8fe9-8a828e08ee0a
# ╟─54fa3a01-5a8d-4dd3-a521-a60a2f69b541
# ╠═d9c33b8c-6ccf-462f-b5c2-267fd7b93ba3
# ╟─2f14fb93-91bc-44af-8a6d-cebaa1607a51
# ╠═fd53c66d-ba90-4d96-9a11-eabc1dd666dc
# ╠═19546bad-d7be-4c09-a248-f7e98c53d42d
# ╟─33c156b6-5c18-4377-9920-32c7865898e9
# ╟─0cdf07a5-a87e-449e-ab09-871924cee6d1
# ╠═0a729667-599a-41f1-98f3-ff94608c15bb
# ╠═0d78c765-1912-4ded-a5e8-aa98afe73cb4
# ╠═5566b0c6-ac96-4342-a62b-0c292469c2eb
# ╟─3bed0c72-ab5c-4dbf-85a5-6f6d062dec97
# ╠═d32536af-b369-4b39-87f6-4e1172d971ae
# ╟─35cdd4f9-87f7-408e-a3fd-7456b7238dd5
# ╠═1320b506-d378-4fa1-bbf1-c2f994abfff1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ db61ecb1-6ed9-44d9-b8e4-b50622efbac1
using TestImages, ImageShow, Random, PlutoTest, Plots, FourierTools, Statistics, FFTW, IndexFunArrays, SpecialFunctions, PlutoUI, Noise, Optim, ForwardDiff, ImageIO, ImageShow

# ╔═╡ fbc3ec99-0c73-40e7-bd21-933bb5a52755
begin
	
	"""
	    simshow(arr; set_one=false, set_zero=false,
	                f=nothing, γ=1)
	Displays a real valued array . Brightness encodes magnitude.
	Works within Jupyter and Pluto.
	# Keyword args
	The transforms are applied in that order.
	* `set_zero=false` subtracts the minimum to set minimum to 1
	* `set_one=false` divides by the maximum to set maximum to 1
	* `f` applies an arbitrary function to the abs array
	* `γ` applies a gamma correction to the abs 
	* `cmap=:gray` applies a colormap provided by ColorSchemes.jl. If `cmap=:gray` simply `Colors.Gray` is used
	    and with different colormaps the result is an `Colors.RGB` element type
	"""
	function simshow(arr::AbstractArray{<:Real}; 
	                 set_one=true, set_zero=false,
	                 f = nothing,
	                 γ = 1,
	                 cmap=:gray)
	    arr = set_zero ? arr .- minimum(arr) : arr
	
	    if set_one
	        m = maximum(arr)
	        if !iszero(m)
	            arr = arr ./ maximum(arr)
	        end
	    end
	
	    arr = isnothing(f) ? arr : f(arr)
	
	    if !isone(γ)
	        arr = arr .^ γ
	    end
	
	
	    if cmap == :gray
	        Gray.(arr)
	    else
	        get(colorschemes[cmap], arr)
	    end
	end
	
	
	"""
	    simshow(arr)
	Displays a complex array. Color encodes phase, brightness encodes magnitude.
	Works within Jupyter and Pluto.
	# Keyword args
	The transforms are applied in that order.
	* `f` applies a function `f` to the array.
	* `absf` applies a function `absf` to the absolute of the array
	* `absγ` applies a gamma correction to the abs 
	"""
	function simshow(arr::AbstractArray{T};
	                 f=nothing,
	                 absγ=1,
	                 absf=nothing) where (T<:Complex)
	
	    if !isnothing(f)
	        arr = f(arr)
	    end
	
	    Tr = real(T)
	    # scale abs to 1
	    absarr = abs.(arr)
	    absarr ./= maximum(absarr)
	
	    if !isnothing(absf)
	        absarr .= absf(absarr)
	    end
	    
	    if !isone(absγ)
	        absarr .= absarr .^ absγ
	    end
	
	    angarr = angle.(arr) ./ Tr(2pi) * 360 
	
	    HSV.(angarr, one(Tr), absarr)
	end
	
	
	
	"""
	    simshow(arr::AbstractArray{<:Colors.ColorTypes.Colorant})
	If `simshow` receives an array which already contains color information, just display it.
	"""
	function simshow(arr::AbstractArray{<:Colors.ColorTypes.Colorant})
	    return arr
	end
end

# ╔═╡ 82726a23-060c-4c7d-91c1-e78a22fb98d6
md"# 1 Fourier Transforms

In the last homework we learnt that FFTW calculates the Fourier transform such that the DC frequency (zero frequency) is at the first entry of the array.

To shift the frequency to the center, we normally use `fftshift`.
However, another issue is, that the FFT also interpretes the center of the array at the first index position.
So for example, we would expect that the Fourier transform of the delta peak is a constant array! 

As you can see below, the Fourier transform is only a constant array if the delta peak is at the fist entry
"

# ╔═╡ ebe90686-c72f-4a42-a644-f06babe00f41
# FFT of delta peak results in constant array
fft([1.0, 0.0, 0.0])

# ╔═╡ 5053c100-5f27-4f2f-acbe-e12c9070feac
# no constant array since delta is not at the first entry
fft([0.0, 1.0, 0.0])

# ╔═╡ 361a7a34-d70b-4597-948c-80ed913a562b
# no constant array since delta is not at the first entry
fft([0.0, 0.0, 1.0])

# ╔═╡ 653a94c9-cb89-47df-8ba5-0ce683440b78
md"
The reason for this behaviour is that the second example is a shifted array of the first one. Hence, in Fourier space this corresponds to a phase ramp!

In the last homework we programmed `ffts(x) = fftshift(fft(x))` and `iffts(x) = ifft(ifftshift(x))`.

We now also introduce `my_ft` and `my_ift`.


The qualitative meaning of the different conventions for the Fourier transform of a signal with length `N` is:
* `fft`: center in real space is at `(1,)` and in Fourier space at `(1,)`
* `ffts`: center in real space is at `(1,)` and in Fourier space at `(N ÷ 2 + 1,)`
* `ft`: center in real space is at `(N ÷ 2 + 1,)` and in Fourier space at `(N ÷ 2 + 1,)`


"

# ╔═╡ b65e0745-82c6-48e1-a956-fa71583dad15
begin
	my_ft(x) = fftshift(fft(ifftshift(x)))
	my_ift(x) = fftshift(ifft(ifftshift(x)))
end

# ╔═╡ c53325bd-eeb6-4663-867d-44fd3ace273b
md"## Task 1.1
Try to change the following test below.
You are only allowed to insert `fftshift` and `ifftshift` statements.
Don't change the order of the Fourier transform calls.
"

# ╔═╡ b57917e8-3d2e-433b-84df-6129f760f954
begin
	arr_even = [1.0,2,3,4]
	arr_odd = [1.0,2,3,4,5]
end

# ╔═╡ 6e0e223d-9107-4121-880a-8ed5d7e5e9c2
md"The Test is broken but we fix it by changing the right hand side. Do that for the red tests below as well"

# ╔═╡ 5213f82b-5e06-4cee-a0d1-21f1c5cc2998
PlutoTest.@test ffts(arr_even) ≈ fftshift(fft(arr_even))

# ╔═╡ 64103e19-ce79-48b7-bac9-b429f6d423c2
# this one is fixed
PlutoTest.@test ffts(arr_even) ≈ fftshift(fft(arr_even))

# ╔═╡ f380db27-8773-413b-978f-4496fc585ae3
md"### Task 1.1
Now try to fix always the right hand side accordingly!
"

# ╔═╡ 0464d972-fcaf-4f83-be7d-073d217e8f4c
# TODO
PlutoTest.@test ift(arr_odd) ≈ fftshift(iffts(arr_odd))

# ╔═╡ 87fabbfe-e25b-467a-9fbd-ceec98ba4ed6
# TODO
PlutoTest.@test iffts(arr_odd) ≈ ifft(ifftshift(arr_odd))

# ╔═╡ c537e21b-5ed2-4cac-947a-e052474a2442
# TODO
PlutoTest.@test ffts(arr_odd) ≈ ft(fftshift(arr_odd))

# ╔═╡ 3093ae8d-27d4-4d32-8021-80fc6b0d1472
# TODO
PlutoTest.@test ifft(arr_odd) ≈ ifftshift(ift(fftshift(arr_odd)))

# ╔═╡ a43b5bb5-d56b-43c7-aa96-ed2d9f217373
md"# 2 Convolution

From calculus class we know that a convolution can be expressed via Fourier transforms like

$U * h = \mathcal{F}^{-1}\left[\mathcal{F}[U] \cdot \mathcal{F}[h] \right] = \mathcal{F}^{-1}\left[\mathcal{F}[U] \cdot H \right]$

where $*$ is a convolution and $\cdot$ an elementwise multiplication. $H$ is the OTF of the PSF $h$.

Now implement it yourself!
"

# ╔═╡ a0362131-41ad-4151-b851-d75af22791e1
"""
	my_conv(U, h)

Calculates a FFT based convolution between U and h.
The output is a real array! 
So either use real valued transforms `rfft` or use `fft` with a final `real` conversion.

# Example
```julia-repl
julia> my_conv([1.0,0,0,0], [1.0, 0.5, 0.0, 0.5])
4-element Vector{ComplexF64}:
 1.0 + 0.0im
 0.5 + 0.0im
 0.0 + 0.0im
 0.5 + 0.0im
```
"""
function my_conv(U::AbstractArray{T, N}, h; center=ntuple(i -> 1, N)) where {T, N}
	# TODO
	h = circshift(h, 1 .- center) #shift the h element to first then conv
	freU = fft(U)
	freH = fft(h)
	conImage = ifft(freU .* freH)

	return real(conImage)
end

# ╔═╡ b568e681-7eaf-4c1a-9f36-f46163c35041
begin
	img = Float32.(testimage("mandril_gray"))
	Gray.(img)
end;

# ╔═╡ cdf36a7b-54ff-4ab7-9497-81d1ed684be3
# some simple kernel
kernel = IndexFunArrays.normal(img, sigma=7);

# ╔═╡ 0d9e2e69-e51f-483c-aadd-6bde1ab0c28a
md"You should see the blurry monkey here. If not, it might be wrong."

# ╔═╡ 5436a802-9d78-48ff-af79-2cf8065fd514
simshow(my_conv(img, kernel, center=(257, 257)))

# ╔═╡ 2eb0bea1-b6dc-417f-81b3-435fede2aa66
md"## 2 Test"

# ╔═╡ c7a4ce47-9bdf-46f3-88a4-888df19b9cb1
PlutoTest.@test my_conv([1.0,2,3,4], [1.0,0,0,0]) ≈ [1,2,3,4]

# ╔═╡ 3b479c44-4f6d-41ad-af70-04616a64154c
PlutoTest.@test my_conv([1.0,2,3,4], [0.0,1.0,0,0], center=2) ≈ [1,2,3,4]

# ╔═╡ 39807680-27cf-444d-9d93-d845baab61a4
PlutoTest.@test my_conv([1.0,2,3,4,5], [0.0,0.0,1.0,0,0], center=3) ≈ [1,2,3,4, 5]

# ╔═╡ 51bcf5d5-0405-467e-b528-275965fa1287
PlutoTest.@test my_conv(img, IndexFunArrays.delta(img, offset=(1,1))) ≈ img

# ╔═╡ da0d4ab1-b1d6-4817-ad38-6d3213c4e075
PlutoTest.@test my_conv([1, 2, 3, 4, 5, 6], [-1, 2, 2, 1, 3, 1]) ≈ [36.0, 32.0, 28.0, 30.0, 20.0, 22.0]

# ╔═╡ 09321d84-f9a7-412d-8fe4-3ece1bd90b21
md"# 3 Incoherent Image Formation

In this part we want to simulate an incoherent imaging situation (fluorescent microscope).
We simplify it by only considering an unitless `radius`.

As reminder, the qualitative procedure is as following:

We take a point source, Fourier transform it, take only the frequency inside the radius, go back to real space, take the absolute squared.

Pay attention where the frequencies should be located when you apply the `circ`.
"

# ╔═╡ ebfe7f3a-098a-41d8-a5bd-2c1e2b374fe0
begin
	"""
		circ(size, radius)
	
	`size` the size of the resulting array
	and `radius` the radius of the circle
	"""
	circ(size, radius) = rr2(size) .<= radius^2
end

# ╔═╡ 772fe3f9-6db8-4df4-8813-cbb334035351
"""
	calc_psf(arr_size, radius)

Calculate a simple PSF. `arr_size` is the output size and `radius` is the radius 
in Fourier space.
The output array is normalized by its sum to 1.
"""
function calc_psf(arr_size, radius)
	# TODO
	aperture  = circ(arr_size, radius)
	# size(img) .÷ 2.0 .+ 1 center point tuple, integer div avoid float
	pointSr = zeros(arr_size)
	pointSr[arr_size[1] ÷ 2 + 1,arr_size[2] ÷ 2 + 1] = 1.0
	
	pointSrFre = fftshift(fft(pointSr))
	pointSrFreC = pointSrFre .* aperture

	psf = abs2.(ifft(ifftshift(pointSrFreC)))
	return psf ./ sum(psf)
end

# ╔═╡ 99154efd-cec4-409a-ae43-bc79c214da1c
#img[(size(ones(11,11)).÷ 2 .+1)...] []can t access tuple, ... some how cancel the block of tuple, make it as a normal array.

# ╔═╡ 33b3a409-2690-4a4a-a091-cdfe6a831c59
md"r = $(@bind r PlutoUI.Slider(0.01:0.1:10.0, show_value=true))"

# ╔═╡ f209c5de-c2bd-4a0b-bbfd-6831e0254023
simshow(calc_psf((64, 64), r))

# ╔═╡ 8b922c48-7d56-4e5b-b039-789e281c5fe1
md"r2 = $(@bind r2 PlutoUI.Slider(1:1:256, show_value=true))"

# ╔═╡ 119da3f0-0f1d-4a65-93cd-868f3c1d5f3e
h = calc_psf(size(img), r2);

# ╔═╡ 5830d782-67a3-4cae-847c-bdbdf0217aa7
# change this line such that the monkey is correctly centerd
# TODO
simshow(my_conv(img, h, center=size(h) .÷ 2 .+ 1))

# ╔═╡ fbdcb172-b079-46a5-b8f3-f5ece30fe25a
md"## 3 Test"

# ╔═╡ 66db7400-9791-4952-9812-39b22829b29a
# large radius is a perfect optical system -> delta peak
PlutoTest.@test calc_psf((2, 2), 1000) ≈  [0 0; 0 1]

# ╔═╡ baff7f36-d18a-4434-b979-662b6d44eb46
PlutoTest.@test sum(calc_psf((13, 12), 3)) ≈ 1

# ╔═╡ 7d92a5c5-d48d-429b-85fa-63904f21fc62
PlutoTest.@test minimum(calc_psf((13, 12), 3)) ≥ 0 

# ╔═╡ fd85fd4a-3e43-431c-aaa6-e92848c9e304
# compare to (approx) analytical solution
begin
	h2 = jinc.(7.25 / 9.219π * IndexFunArrays.rr((64, 64))).^2
	h2 ./= sum(h2)
	PlutoTest.@test ≈(1 .+ h2, 1 .+ calc_psf((64, 64), 7.25), rtol=0.001)
end

# ╔═╡ 08936039-7557-4ea3-8e26-5fbcdf12aec2
md"# 4 Generalized Wiener Filtering

A simpled deconvolution approach suited for Gaussian noise is the Wiener filter.
You can find the details in the slides.
Try to implement it here!
"

# ╔═╡ 7256f510-10ac-4fb2-99fc-0ffbcd45aae3
function wiener_filter(img, h, ϵ)
	# TODO
    imgFre = fft(img)
    hFre = fft(h)

    wienerFilterFre = conj(hFre) ./ (abs2.(hFre) .+ ϵ)  # Wiener filter formula

    restoredFre = imgFre .* wienerFilterFre

    restoredImg = real(ifft(restoredFre))

    return restoredImg
end

# ╔═╡ 9f4d5c69-c5a8-4a34-9100-e8209ede71b4
begin
	# PSF
	h3 = ifftshift(calc_psf(size(img), 20))
	simshow(h3)
	# attente the fftshift ifftshift problem
end

# ╔═╡ 2d5653cf-fa36-4a0e-99cd-4c9769b84705
begin
	img_b = my_conv(img, h3)
	simshow(img_b)
end

# ╔═╡ 0f8555c7-a1c2-4643-912b-6816019a848a
img_gauss = add_gauss(img_b, 0.1);

# ╔═╡ 34ced1ed-82f4-4c32-9cec-086ff2a62bce
img_poisson = poisson(img_b, 20);

# ╔═╡ cbd3d6c4-bf40-4431-a7b9-5b26b411a9b9
simshow([img_gauss img_poisson])

# ╔═╡ 29cc8e86-c109-411e-ba86-a8155f7c3a94
md"
pow1 = $(@bind pow1 Slider(-6:0.1:-0, show_value=true))

pow2 = $(@bind pow2 Slider(-6:0.1:-0, show_value=true))
"

# ╔═╡ 82542705-e24e-409e-a7bd-491ce750007e
img_gauss_wiener = wiener_filter(img_gauss, h3, 10^pow1);

# ╔═╡ e50febee-5c4b-44d7-9d78-29395a8c3ab6
img_poisson_wiener = wiener_filter(img_poisson, h3, 10^pow2);

# ╔═╡ 0b46da45-b743-40a1-a1f8-0581b6fd741a
simshow([img_gauss_wiener  img_poisson_wiener])

# ╔═╡ f85f964d-0443-4e19-b655-036f82a0ba69
md"## 4 Test"

# ╔═╡ 94f64888-b732-4367-a905-b57de684fcf7
PlutoTest.@test wiener_filter([1.0, 2.0], [1.0, 0.0], 0) ≈ [1.0, 2.0]

# ╔═╡ 3d89c0be-a888-4e2f-8820-85e07bd6be30
PlutoTest.@test  wiener_filter([1.0, 2.0], [1.0, 0.1], 0)  ≈ [0.808081, 1.91919] rtol=1e-5

# ╔═╡ c1e8419c-c748-490b-93d3-9bbcde4f4da9
md"# 5 Gradient Descent Optimization
In this part we want to implement an optimization routine with a _strange_ sensor.

The sensor has an additive Gaussian noise part but also an cubic gain behaviour.
See the function below
"

# ╔═╡ c337e8cf-bab9-46cd-aae8-22f6265f9cb7
begin
	function strange_sensor_f(value::T) where T
		value .^2
	end

	function strange_sensor(value::T) where T
		return abs.(randn(T) * 0.10f0 + strange_sensor_f(value))
	end
end

# ╔═╡ ff2288f0-2edf-4f22-90ef-af5777676ae7
# strange output
strange_sensor.([1.0 2; 3 4])

# ╔═╡ d23fe57a-7255-4d8f-a00a-eaec953213b2
md"
First we simulate the full `img` of the mandril with that sensor.
Clearly the appearance has changed due to the quadratic behaviour but also noise is visible.
"

# ╔═╡ a0f77be2-c40c-4372-a245-45b76ddc5861
begin
	img_strange = strange_sensor.(img) # todo
	simshow(img_strange)
end

# ╔═╡ cdc25e03-c6fd-4ee0-801e-24654a83d65d
md"## 5.1 Task
Since our sensor is quite noisy, we want to measure the image multiple times.
Try to complete `measure`.
"

# ╔═╡ 4af883c2-fdc9-458c-998b-ff0b8b5df146
"""
	measure(img, N)

Measure the `img` `N` times. Return a vector with the `N` measured images.
"""
function measure(img, N)
	# TODO
	return [strange_sensor.(img) for i in 1:N]
end

# ╔═╡ c02969bc-3b5e-4422-a3c9-3a7d00cb3ddf
imgs = measure(img, 10); # TODO: simulate 10 times

# ╔═╡ e3a8ef81-0d19-4e75-b137-6454ce262991
simshow([reduce(hcat, imgs[1:end÷2]); reduce(hcat, imgs[end÷2+1:end])])

# ╔═╡ ebd74bcb-4187-469f-a206-ecf9041918f1
md"## 5.1 Test"

# ╔═╡ 7579c331-fa53-47f4-a479-312f2f7a3931
PlutoTest.@test typeof(measure([1.0], 12)) <: Vector

# ╔═╡ 3cca079c-1245-493b-ae6c-9db18e11835c
PlutoTest.@test length(measure([1.0], 12)) == 12

# ╔═╡ 0ae38ccc-9129-4c61-9810-dbc448e8a07f
begin
	Random.seed!(42)
	a = measure([1.0 2.0; 3 4], 3)
end;

# ╔═╡ 4f5a9a27-1893-49e4-a2e9-59ba3a2340a2
begin
		Random.seed!(42)
		b = [abs.([1.0 2.0; 3 4].^2 .+ randn((2,2)) * 0.1f0) for i = 1:3]
end;

# ╔═╡ 37ce2f96-377d-48d7-bc56-f83b0cce349c
PlutoTest.@test a ≈ b

# ╔═╡ cd0829a1-cdf1-45c9-a4dd-91bb3ec5bb03
md"## 5.2 Loss Function
Having our 10 images we would like to retrieve a best guess for the underlying image.
Taking the mean does not work in this case since the input image is modified by `strange_sensor_f`.

Therefore, we interprete the reconstruction as an optimization problem.
Try to find the corresponding pages in the lecture.

Generally, we want to minimize a loss function which looks like

$\underset{\mu}{\mathrm{argmin}}\, \mathcal L = \sum_{\text{img} \, \in \, \text{imgs}} \sum_{p_i \, \in \, \text{img}} (\text{img}[i] - f(\mu[i]))^2$


So we sum over all measured images. For each image, we additionally sum each pixel and  compare it with $f(\mu[i])$.
$f$ is the same as `strange_sensor_f` and by calling $f(\mu[i])$ we apply $f$ to the reconstruction $\mu$. Via that function call, we hope that we find a $\mu$ which fits to the measurments. So $f$ is the forward model of the sensor (without the noise part).
"

# ╔═╡ 328b5994-b1c4-4850-b8d3-2f3781bed99f
"""
	loss(imgs, μ)

Calculate the loss between the `ìmgs` and `μ`.
Basically implement the sum of the square difference value.
Don't forget to apply `strange_sensor_f` to `μ` before!

Using two for loops is perfectly fine!
"""
function loss(imgs::Vector{<:AbstractArray{T, N}}, μ::AbstractArray{T, N}) where {T, N}
	# TODO
	#calculate the L2 norm
	imgLoss = Vector{T}(undef,length(imgs))
	for ii in 1:length(imgs)
		imgLoss[ii] = sum(abs2.(imgs[ii] .- strange_sensor_f.(μ)))
	end
	lossValue = sum(imgLoss)
	#simple way:
	# for img in imgs
	# 	img += sum(abs2.(img .- strange_sensor_f))
	# end
	return lossValue
end

# ╔═╡ 78da1c47-1de7-4d95-a1e1-cd60e912a0fc
# comparison with ground truth image
loss(imgs, img) 

# ╔═╡ 3d182fa9-84bb-4504-8945-bf653ce1f99d
# the ground truth is not the minimum...
# why not?
loss(imgs, img .+0.0001f0) 

# ╔═╡ b56a311e-912a-4e7c-b821-4124b847194a
md"## 5.2 Test"

# ╔═╡ 840ddb7e-56ba-4d7a-8f08-cee67128315f
PlutoTest.@test loss([[1]], [2]) isa Number

# ╔═╡ 528194f2-9f74-4dda-94f8-3e4f0e515bc9
PlutoTest.@test loss([[1]], [2])  ≈ 9

# ╔═╡ a203fb4f-c3d6-4d46-9c58-2454b71b0e1b
PlutoTest.@test loss([[1], [2], [4.0]], sqrt.([2]))  ≈ 5

# ╔═╡ 063ef53f-d152-4282-af5f-7f2addce3ab0
md"## 5.3 Gradient of `strange_sensor_f`
For the optimization we later want to apply a gradient descent optimization scheme.
Hence, we need a few gradients!
"

# ╔═╡ 95392d70-07cf-47a7-8707-f1dab177e7a5
"""
	 grad_strange_sensor_f(value::T)

Calculate the gradient of `strange_sensor_f`.
Use the rules you know already from school!
"""
function grad_strange_sensor_f(value::T) where T
	# TODO
	grad = 2 .* value
	return grad
end

# ╔═╡ 01de98fe-5537-4745-acf8-99daa0b8aa6f
grad_strange_sensor_f(1)

# ╔═╡ 75549871-14a3-4b9f-9b98-cccc34ed315d
md"## 5.3 Test"

# ╔═╡ 867b3bf0-1208-48a1-b1ee-904d11b28e1f
PlutoTest.@test grad_strange_sensor_f(0) ≈ 0

# ╔═╡ 7c2d2d6d-f933-452e-9870-7dce7ad4bf1d
# comparison with automatic differentation package!
PlutoTest.@test ForwardDiff.derivative(strange_sensor_f, 42) ≈ grad_strange_sensor_f(42)

# ╔═╡ cd363a92-a473-47f8-b41b-c5d5249fec90
md"## 5.4 Gradient of Loss
Now we come to the last part of the gradient.
We need the full gradient of the loss function.

The gradient of the loss function with respect to `μ` will be an array again.
That is plausible since the loss function accounts for all pixels. By changing a single pixel we also change the value of the loss function. Therefore, for each pixel a gradient exists describing the influence to the loss value.

You need to derive the loss with respect to $\mu$. But don't forget about the chain rule for `strange_sensor_f`!

Again, two for loops are perfectly fine!
"

# ╔═╡ 1ad4cf95-a3fb-4e45-b7bd-80ab724c158e
function gradient(imgs::Vector{<:AbstractArray{T, N}}, μ::AbstractArray{T, N}) where {T, N}
	# TODO
	# chain rule for L（img-f(μ))=2*(img-f(μ))*(2μ)
    grad = zeros(size(μ))
    for img in imgs
		# must be estimate value - observed value to guarantee the gradient direction
		# there is a - in front of the μ, so should be - in front of the gradient.
        sensor_diff = strange_sensor_f.(μ) .- img 
        for idx in eachindex(μ)
            grad[idx] += 2 * sensor_diff[idx] * grad_strange_sensor_f(μ[idx])
        end
    end
    
    return grad
end

# ╔═╡ cb41a2f9-a0c5-4eeb-af51-9bf40a5d7fd6
gradient(imgs, img)

# ╔═╡ f72450cb-a54e-4827-9c33-7a44f561bd43
md"## 5.4 Test"

# ╔═╡ 7e6eb33d-0721-4bb6-a9ee-1cfb07a17e46
PlutoTest.@test gradient([[1.0], [2.0]], [2.5]) ≈ [95]

# ╔═╡ cf550002-13ce-4788-8206-790fb355a91b
PlutoTest.@test gradient([[3.5, 2.1], [2.0, 0.0]], [3.23, 23.2]) isa Vector

# ╔═╡ bfd35c9f-1449-4d86-bb50-9bbbd6ea2c30
PlutoTest.@test size(gradient([[3.5, 2.1], [2.0, 0.0]], [3.23, 23.2])) == (2,)

# ╔═╡ c1795dd9-f8a5-472d-bd82-7871e8603534
PlutoTest.@test  gradient([[3.5, 2.1], [2.0, 0.0]], [3.23, 23.2]) ≈ [198.526136, 99702.46399999999]

# ╔═╡ c5436e99-513b-4c51-b3ad-f82318304a3e
md"## 5.5 Gradient Descent"

# ╔═╡ b393a0f5-af03-4ded-8e22-886022f5da30
"""
	gradient_descent(imgs, μ, N_iter, step_size)

Runs a gradient descent using `gradient` and the experimental images `imgs`.
`N_iter` is the number of iterations, `step_size` is the step size of the gradient step.

The optimized output is `μ_optimized`.
"""
function gradient_descent(imgs, μ, N_iter, step_size)
	# TODO
	μ_optimized = copy(μ)
	for i = 1:N_iter
		μ_optimized .= μ_optimized .- step_size .* gradient(imgs, μ_optimized)
	end
	return μ_optimized
end

# ╔═╡ c81c5eaa-0e0f-4730-9148-ec0e7b63cdd4
μ_init = mean(imgs) # TODO, what could be a good initilization?

# ╔═╡ 32f6d83a-d5db-4a21-b2c4-d8876de38c46
md"
Try to change the gradient step and the number of iterations.

Number of iterations $(@bind N_iter Slider(1:100, show_value=true))

step size $(@bind step_size Slider(1f-5:1f-5:1f-1, show_value=true))
"

# ╔═╡ f7d7cf86-ece9-406b-af78-f86177a767c4
μ_optimized = gradient_descent(imgs, μ_init, N_iter, step_size);

# ╔═╡ 2110469f-16f9-4454-996a-a949a94fffa3
# that value should get smaller with more iterations
# smaller is better
loss(imgs, μ_optimized)
# iter 100, step size 0.02743 minimum, beat the mean and optimal

# ╔═╡ 3643e55e-3205-4561-b271-058d59f46685
# this value should be larger than ↑
loss(imgs, μ_init)

# ╔═╡ 85c81f80-c959-4682-932b-26fe7f415f4d
Gray.(μ_optimized)

# ╔═╡ 09313c1d-102f-4e0b-b9c8-7d2e904a1dd6
md"## 5.5 Test"

# ╔═╡ b2e7aae8-2777-4657-ba3d-f0dfcb581ede
PlutoTest.@test gradient_descent([[1.0], [2.0]], [3.0], 3, 0.01) ≈ [1.21021] rtol=1f-4

# ╔═╡ ae027113-998b-467e-8967-cc416d480377
PlutoTest.@test loss(imgs, μ_init) > loss(imgs, μ_optimized)

# ╔═╡ 4ff0afae-af4c-45d0-83b2-4fb5eead2fa1
PlutoTest.@test loss(imgs, gradient_descent(imgs, (imgs[1] .+ imgs[2]) / 2, 15, 0.000001);) < loss(imgs, imgs[1])

# ╔═╡ 2374bfc1-cd14-4bee-b5e5-bb15e0e6e253
begin
	μ_mean = sqrt.(reduce((a,b) -> a .+ b, imgs) ./ 10)
	Gray.(μ_mean)
end;

# ╔═╡ 9360c9a0-6a0f-4ff6-a215-23ce5c3c8099
md"#### Can you beat the _mean_?"

# ╔═╡ a837b22c-61f1-4a98-a362-c08b405c4dca
PlutoTest.@test loss(imgs, μ_optimized) < loss(imgs, μ_mean)

# ╔═╡ e2aec734-ce2b-4761-b3d0-b559df8b17da
md"Note that in this particular example the mean can be proven to be the best estimator, so it is not surprising to not be able to beat the mean. Yet, if the model cannot be inverted, we still may want to use a gradient-based optimization to solve the problem."

# ╔═╡ 959b4745-7742-459a-bef0-e374ec4aec17
[simshow(img) simshow(μ_mean) simshow(μ_optimized)]

# ╔═╡ b3d54170-466c-4b0d-a337-280ef4ea87f3
md"## 5.6 Test
To check wether your gradient and your loss is correct, see also the following output of Optim (a sophisticated package for optimization).
If the output is not a nice Mandrill, there is most likely something wrong with your loss/gradients.
"

# ╔═╡ eafe5f6e-18ab-4341-8435-a3c0b9423b35
begin
	g_optim!(G, x) = isnothing(G) ? nothing : G .= gradient(imgs, x)
	f_optim(x) = loss(imgs, x)
end

# ╔═╡ 13304ea6-0639-4863-827e-d66a8630bc63
begin
	μ_optim_init = copy(μ_init)
	res = optimize(f_optim, g_optim!, μ_optim_init, ConjugateGradient(), Optim.Options(iterations=50))
	μ_optim = Optim.minimizer(res)
	res
end

# ╔═╡ c685ac28-79e8-4eda-8c0f-0944a1276691
simshow(Optim.minimizer(res))

# ╔═╡ 7a691c80-e08d-423a-a69a-572f02fdddda
loss(imgs, μ_optim)

# ╔═╡ 19a0e512-46d6-429e-93f0-05e2faf95218
md"## Can you beat Optim?"

# ╔═╡ abe5aca2-61ce-4f99-ad75-0c84293cd7b3
PlutoTest.@test loss(imgs, μ_optimized) < loss(imgs, μ_optim)

# ╔═╡ 1171004b-96a1-4ee4-bbd5-2b7f2b1d582a
# yes, but hard with an hand optimized iterations/value pair

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
FourierTools = "b18b359b-aebc-45ac-a139-9c0ccbb2871e"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
IndexFunArrays = "613c443e-d742-454e-bfc6-1d7f8dd76566"
Noise = "81d43f40-5267-43b7-ae1c-8b967f377efa"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
FFTW = "~1.5.0"
ForwardDiff = "~0.10.33"
FourierTools = "~0.3.7"
ImageIO = "~0.6.6"
ImageShow = "~0.3.6"
IndexFunArrays = "~0.2.5"
Noise = "~0.3.2"
Optim = "~1.7.4"
Plots = "~1.36.6"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.49"
SpecialFunctions = "~2.1.7"
TestImages = "~1.7.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "eed75f377c1c0cccf6b70ac1c218877c54cf8509"

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

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "0ba8f4c1f06707985ffb4804fdad1bf97b233897"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.41"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

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

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

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

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

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

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

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

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

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

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

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
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FourierTools]]
deps = ["ChainRulesCore", "FFTW", "IndexFunArrays", "LinearAlgebra", "NDTools", "NFFT", "PaddedViews", "Reexport", "ShiftedArrays"]
git-tree-sha1 = "3cbae3e75b991e2b58034af6de389a6d26e33cfe"
uuid = "b18b359b-aebc-45ac-a139-9c0ccbb2871e"
version = "0.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "4423d87dc2d3201f3f1768a29e807ddc8cc867ef"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3657eb348d44575cc5560c80d7e55b812ff6ffe1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

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
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

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
git-tree-sha1 = "4d5fc006e0a006875f57be883c81d9c4a5d56bc6"
uuid = "98581153-e998-4eef-8d0d-5ec2c052313d"
version = "0.5.3"

[[deps.NFFT]]
deps = ["AbstractNFFTs", "BasicInterpolators", "Distributed", "FFTW", "FLoops", "LinearAlgebra", "PrecompileTools", "Printf", "Random", "Reexport", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "d51436bbf2dfd3f8b89de773b661759e89f24050"
uuid = "efe261a4-0d2b-5849-be55-fc731d526b0d"
version = "0.13.6"

    [deps.NFFT.extensions]
    NFFTGPUArraysExt = ["Adapt", "GPUArrays"]

    [deps.NFFT.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

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
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

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

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ad31332567b189f508a3ea8957a2640b1147ab00"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

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

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6a9521b955b816aa500462951aa67f3e4467248a"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.36.6"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PoissonRandom]]
deps = ["Random"]
git-tree-sha1 = "a0f1159c33f846aa77c3f30ebbc69795e5327152"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.4"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

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

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

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

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

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

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

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

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

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
git-tree-sha1 = "03492434a1bdde3026288939fc31b5660407b624"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.7.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "f21231b166166bebc73b99cea236071eb047525b"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.3"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

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
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "ee6f41aac16f6c9a8cab34e2f7a200418b1cc1e3"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+0"

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
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "055a96774f383318750a1a5e10fd4151f04c29c5"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.46+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╠═db61ecb1-6ed9-44d9-b8e4-b50622efbac1
# ╠═fbc3ec99-0c73-40e7-bd21-933bb5a52755
# ╟─82726a23-060c-4c7d-91c1-e78a22fb98d6
# ╠═ebe90686-c72f-4a42-a644-f06babe00f41
# ╠═5053c100-5f27-4f2f-acbe-e12c9070feac
# ╠═361a7a34-d70b-4597-948c-80ed913a562b
# ╟─653a94c9-cb89-47df-8ba5-0ce683440b78
# ╠═b65e0745-82c6-48e1-a956-fa71583dad15
# ╟─c53325bd-eeb6-4663-867d-44fd3ace273b
# ╠═b57917e8-3d2e-433b-84df-6129f760f954
# ╟─6e0e223d-9107-4121-880a-8ed5d7e5e9c2
# ╠═5213f82b-5e06-4cee-a0d1-21f1c5cc2998
# ╠═64103e19-ce79-48b7-bac9-b429f6d423c2
# ╟─f380db27-8773-413b-978f-4496fc585ae3
# ╠═0464d972-fcaf-4f83-be7d-073d217e8f4c
# ╠═87fabbfe-e25b-467a-9fbd-ceec98ba4ed6
# ╠═c537e21b-5ed2-4cac-947a-e052474a2442
# ╠═3093ae8d-27d4-4d32-8021-80fc6b0d1472
# ╟─a43b5bb5-d56b-43c7-aa96-ed2d9f217373
# ╠═a0362131-41ad-4151-b851-d75af22791e1
# ╠═b568e681-7eaf-4c1a-9f36-f46163c35041
# ╠═cdf36a7b-54ff-4ab7-9497-81d1ed684be3
# ╟─0d9e2e69-e51f-483c-aadd-6bde1ab0c28a
# ╠═5436a802-9d78-48ff-af79-2cf8065fd514
# ╟─2eb0bea1-b6dc-417f-81b3-435fede2aa66
# ╠═c7a4ce47-9bdf-46f3-88a4-888df19b9cb1
# ╠═3b479c44-4f6d-41ad-af70-04616a64154c
# ╠═39807680-27cf-444d-9d93-d845baab61a4
# ╠═51bcf5d5-0405-467e-b528-275965fa1287
# ╠═da0d4ab1-b1d6-4817-ad38-6d3213c4e075
# ╟─09321d84-f9a7-412d-8fe4-3ece1bd90b21
# ╠═ebfe7f3a-098a-41d8-a5bd-2c1e2b374fe0
# ╠═772fe3f9-6db8-4df4-8813-cbb334035351
# ╠═99154efd-cec4-409a-ae43-bc79c214da1c
# ╟─33b3a409-2690-4a4a-a091-cdfe6a831c59
# ╠═f209c5de-c2bd-4a0b-bbfd-6831e0254023
# ╟─8b922c48-7d56-4e5b-b039-789e281c5fe1
# ╠═119da3f0-0f1d-4a65-93cd-868f3c1d5f3e
# ╠═5830d782-67a3-4cae-847c-bdbdf0217aa7
# ╟─fbdcb172-b079-46a5-b8f3-f5ece30fe25a
# ╠═66db7400-9791-4952-9812-39b22829b29a
# ╠═baff7f36-d18a-4434-b979-662b6d44eb46
# ╠═7d92a5c5-d48d-429b-85fa-63904f21fc62
# ╠═fd85fd4a-3e43-431c-aaa6-e92848c9e304
# ╟─08936039-7557-4ea3-8e26-5fbcdf12aec2
# ╠═7256f510-10ac-4fb2-99fc-0ffbcd45aae3
# ╠═9f4d5c69-c5a8-4a34-9100-e8209ede71b4
# ╠═2d5653cf-fa36-4a0e-99cd-4c9769b84705
# ╠═0f8555c7-a1c2-4643-912b-6816019a848a
# ╠═34ced1ed-82f4-4c32-9cec-086ff2a62bce
# ╠═cbd3d6c4-bf40-4431-a7b9-5b26b411a9b9
# ╠═29cc8e86-c109-411e-ba86-a8155f7c3a94
# ╠═82542705-e24e-409e-a7bd-491ce750007e
# ╠═e50febee-5c4b-44d7-9d78-29395a8c3ab6
# ╠═0b46da45-b743-40a1-a1f8-0581b6fd741a
# ╟─f85f964d-0443-4e19-b655-036f82a0ba69
# ╠═94f64888-b732-4367-a905-b57de684fcf7
# ╠═3d89c0be-a888-4e2f-8820-85e07bd6be30
# ╟─c1e8419c-c748-490b-93d3-9bbcde4f4da9
# ╠═c337e8cf-bab9-46cd-aae8-22f6265f9cb7
# ╠═ff2288f0-2edf-4f22-90ef-af5777676ae7
# ╟─d23fe57a-7255-4d8f-a00a-eaec953213b2
# ╠═a0f77be2-c40c-4372-a245-45b76ddc5861
# ╟─cdc25e03-c6fd-4ee0-801e-24654a83d65d
# ╠═4af883c2-fdc9-458c-998b-ff0b8b5df146
# ╠═c02969bc-3b5e-4422-a3c9-3a7d00cb3ddf
# ╠═e3a8ef81-0d19-4e75-b137-6454ce262991
# ╟─ebd74bcb-4187-469f-a206-ecf9041918f1
# ╠═7579c331-fa53-47f4-a479-312f2f7a3931
# ╠═3cca079c-1245-493b-ae6c-9db18e11835c
# ╟─0ae38ccc-9129-4c61-9810-dbc448e8a07f
# ╟─4f5a9a27-1893-49e4-a2e9-59ba3a2340a2
# ╠═37ce2f96-377d-48d7-bc56-f83b0cce349c
# ╟─cd0829a1-cdf1-45c9-a4dd-91bb3ec5bb03
# ╠═328b5994-b1c4-4850-b8d3-2f3781bed99f
# ╠═78da1c47-1de7-4d95-a1e1-cd60e912a0fc
# ╠═3d182fa9-84bb-4504-8945-bf653ce1f99d
# ╟─b56a311e-912a-4e7c-b821-4124b847194a
# ╠═840ddb7e-56ba-4d7a-8f08-cee67128315f
# ╠═528194f2-9f74-4dda-94f8-3e4f0e515bc9
# ╠═a203fb4f-c3d6-4d46-9c58-2454b71b0e1b
# ╟─063ef53f-d152-4282-af5f-7f2addce3ab0
# ╠═95392d70-07cf-47a7-8707-f1dab177e7a5
# ╠═01de98fe-5537-4745-acf8-99daa0b8aa6f
# ╟─75549871-14a3-4b9f-9b98-cccc34ed315d
# ╠═867b3bf0-1208-48a1-b1ee-904d11b28e1f
# ╠═7c2d2d6d-f933-452e-9870-7dce7ad4bf1d
# ╟─cd363a92-a473-47f8-b41b-c5d5249fec90
# ╠═1ad4cf95-a3fb-4e45-b7bd-80ab724c158e
# ╠═cb41a2f9-a0c5-4eeb-af51-9bf40a5d7fd6
# ╟─f72450cb-a54e-4827-9c33-7a44f561bd43
# ╠═7e6eb33d-0721-4bb6-a9ee-1cfb07a17e46
# ╠═cf550002-13ce-4788-8206-790fb355a91b
# ╠═bfd35c9f-1449-4d86-bb50-9bbbd6ea2c30
# ╠═c1795dd9-f8a5-472d-bd82-7871e8603534
# ╟─c5436e99-513b-4c51-b3ad-f82318304a3e
# ╠═b393a0f5-af03-4ded-8e22-886022f5da30
# ╠═c81c5eaa-0e0f-4730-9148-ec0e7b63cdd4
# ╟─32f6d83a-d5db-4a21-b2c4-d8876de38c46
# ╠═f7d7cf86-ece9-406b-af78-f86177a767c4
# ╠═2110469f-16f9-4454-996a-a949a94fffa3
# ╠═3643e55e-3205-4561-b271-058d59f46685
# ╠═85c81f80-c959-4682-932b-26fe7f415f4d
# ╟─09313c1d-102f-4e0b-b9c8-7d2e904a1dd6
# ╠═b2e7aae8-2777-4657-ba3d-f0dfcb581ede
# ╠═ae027113-998b-467e-8967-cc416d480377
# ╠═4ff0afae-af4c-45d0-83b2-4fb5eead2fa1
# ╠═2374bfc1-cd14-4bee-b5e5-bb15e0e6e253
# ╟─9360c9a0-6a0f-4ff6-a215-23ce5c3c8099
# ╠═a837b22c-61f1-4a98-a362-c08b405c4dca
# ╟─e2aec734-ce2b-4761-b3d0-b559df8b17da
# ╠═959b4745-7742-459a-bef0-e374ec4aec17
# ╟─b3d54170-466c-4b0d-a337-280ef4ea87f3
# ╠═eafe5f6e-18ab-4341-8435-a3c0b9423b35
# ╠═13304ea6-0639-4863-827e-d66a8630bc63
# ╠═c685ac28-79e8-4eda-8c0f-0944a1276691
# ╠═7a691c80-e08d-423a-a69a-572f02fdddda
# ╟─19a0e512-46d6-429e-93f0-05e2faf95218
# ╠═abe5aca2-61ce-4f99-ad75-0c84293cd7b3
# ╟─1171004b-96a1-4ee4-bbd5-2b7f2b1d582a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

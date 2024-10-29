### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 8a11c2b7-b6e5-4c26-9e06-ad3ab958db9c
using TestImages, ImageShow, Colors, Statistics, PlutoTest, PlutoUI

# ╔═╡ 3ccbf4fb-0707-4ce8-83d8-b1af555f3fe1
using PoissonRandom

# ╔═╡ 62c6e2f0-0007-40ec-81c5-61cda80f59e5
using ImageFiltering, IndexFunArrays

# ╔═╡ 31cc54e9-d47b-4df7-a785-dad0120e8cdb
md"## Load Packages"

# ╔═╡ eb8a1967-7b7a-4ee8-851a-992c9dd13486


# ╔═╡ b731d80c-3012-49f5-8e63-9b7cea0ed667
"""
    my_show(arr::AbstractArray{<:Real}; set_one=false, set_zero=false)
Displays a real valued array . Brightness encodes magnitude.
Works within Jupyter and Pluto.
## Keyword args
* `set_one=false` divides by the maximum to set maximum to 1
* `set_zero=false` subtracts the minimum to set minimum to 1
"""
function my_show(arr::AbstractArray{<:Real}; set_one=true, set_zero=false)
    arr = set_zero ? arr .- minimum(arr) : arr
    arr = set_one ? arr ./ maximum(arr) : arr
    Gray.(arr)
end

# ╔═╡ d8c83f2c-76ea-42b9-bd39-7fa5cafcd4d3
md"# Homework 01"

# ╔═╡ 334ebdc5-dc11-4e83-ad12-a1961420a97b
begin
	img = Float64.(testimage("mandril_gray"))
	my_show(img)
end

# ╔═╡ 7a7e06df-dba4-4a55-81bc-f7129ab7d874
begin
	my_show(img,set_zero=true)
end

# ╔═╡ 2122c12f-0832-4f19-9a77-3047d5311cba
md"## 1. Add Noise To Images

This homework is dedicated to introduce into the basics of Julia.
As examples, we want to apply and remove noise from images
"

# ╔═╡ a7d7a415-7464-43fb-a2bf-8e3f2d999c32
md"## Task 1.1  - Gaussian Noise
In the first step, we want to add Gaussian noise with a certain 
standard deviation $\sigma$. Try to get yourself used to the Julia documentation
and check out whether such a function exists already and how you can apply it
to your image/array.

Codewords: **Julia, random, random normal distribution**
"

# ╔═╡ de104b0a-28a4-42de-ae8d-fa857a8f32e0
"""
	add_gauss_noise(img, σ=1)

This function adds normal distributed noise to `img`.
`σ` is an optional argument
"""
function add_gauss_noise(img, σ=one(eltype(img)))
	# this is still wrong
	# img .+= σ * randn() #only generate one single value

	# correction:
	#img .+= σ .* randn(size(img))
	noiseImg = img .+ σ .* randn(size(img))
	return noiseImg
	#return img
end

# ╔═╡ f89bc1e4-d406-4550-988b-71496b64035a
md"
Now we want to use a for loop inside this function.
Try to find out how to write for loops to iterate through an array.
"

# ╔═╡ c0b5ff08-7342-4c25-9791-be3a4918c400
"""
	add_gauss_noise_fl!(img, σ=1)

This function adds normal distributed noise to `img`.
`σ` is an optional argument.
This function is memory efficient by using for loops.


`!` means that the input (`img`) is modified.
Therefore, don't return a new array but instead modify the existing one!
The bang (!) is a convention in Julia that a function modifies the input.
"""
function add_gauss_noise_fl!(img, σ=one(eltype(img)))
	# todo
	for ii in 1:size(img,1)
		for jj in 1:size(img,2)
			img[ii,jj] += σ .* randn()
		end
	end
	return img
end

# ╔═╡ d56bc500-f5cf-40f3-8c0b-b4ae26a58367
md"### Tests 1.1
Don't modify but rather take those tests as input whether you are correct.
Green is excellent, red not :(
"

# ╔═╡ 1116455e-4698-489a-add2-4e1d8ecbb546
my_show(add_gauss_noise(img, 0.15), set_one=true)

# ╔═╡ cc7bbb55-b8dd-454b-b170-45137ab0c0b5
md"###### Check wether mean and standard deviation are correct"

# ╔═╡ c997c248-fba8-4ed8-ab75-179d248e6b83
PlutoTest.@test ≈(0.3, std(add_gauss_noise(ones((512, 512)), 0.3)), rtol=0.1)

# ╔═╡ 07bc00b7-81ad-4645-8fbe-8545ccdfe7e4
PlutoTest.@test ≈(0.3, std(add_gauss_noise_fl!(ones((512, 512)), 0.3)), rtol=0.1)

# ╔═╡ d317ed1a-1867-4293-8db4-51352212929c
PlutoTest.@test ≈(1, mean(add_gauss_noise(ones((512, 512)), 0.3)), rtol=0.1)

# ╔═╡ e69a8b77-833a-4498-8bc9-e4c970f6529d
md"## Task 1.2 - Poisson Noise
In microscopy and low light imaging situation, the dominant noise term is usually Poisson noise which we want to simulate here.

For adding Poisson noise we use: [PoissonRandom.jl](https://github.com/SciML/PoissonRandom.jl)

Read the documentation of this package how to generate Poisson Random numbers
"

# ╔═╡ 51a7dfc9-bfdc-46da-bba5-dcbee4ba5770
pois_rand(100)

# ╔═╡ 1e4b32cb-00ea-439f-900e-abadc850be95
"""
	add_poisson_noise!(img, scale_to=nothing)

This function adds poisson distributed noise to `img`.

Before adding noise, it scales the maximum value to `scale_to` and 
divides by it afterwards.
With that we can set the number of events (like a photon count).

Differently said: the `scale_to` applies Noise to an array equivalent to the noise level of an array with maximum peak `scale_to`.

If `isnothing(scale_to) == true`, we don't modify/scale the array.

`!` means that the input is modified.
"""
function add_poisson_noise!(img, scale_to=nothing)
	# todo
	if isnothing(scale_to)
		for ii in 1:size(img,1)
			for jj in 1:size(img,2)
				img[ii,jj] = pois_rand(img[ii,jj])
			end
		end
	else
		scaleFactor = scale_to ./ maximum(img)
		img .*= scaleFactor 
		for ii in 1:size(img,1)
			for jj in 1:size(img,2)
				img[ii,jj] = pois_rand(scale_to)
			end
		end
		img ./= scaleFactor
	end
	return img
end

# ╔═╡ 6cf8ff2a-4255-4c53-9376-5a0d2d372569
add_poisson_noise!(10 .* ones((3, 3)))

# ╔═╡ 9a3feb13-0424-448c-934b-45c476cb0096
add_poisson_noise!(10 .* ones((3, 3)), 100)

# ╔═╡ e3fcd381-9652-4e1d-8585-a91f87b73ce5
md"### Test 1.2 - Poisson Noise

"

# ╔═╡ f3efcfcf-4516-43d2-9375-d8863b4b6f5b
my_show(add_poisson_noise!(100 .* img), set_one=true)

# ╔═╡ 5b68030f-557e-449e-8baf-cf7ae65e4425
mean(add_poisson_noise!(150 .* ones(Float64, (512, 512))))

# ╔═╡ 78a39a77-bc9d-427a-a113-4e64b0c9d311
PlutoTest.@test ≈(150, mean(add_poisson_noise!(150 .* ones(Float64, (512, 512)))), rtol=0.05)

# ╔═╡ 41118b3c-3e49-4fc7-8c80-7906b7cf91fe
PlutoTest.@test ≈(√(150), std(add_poisson_noise!(150 * ones(Float64, (512, 512)))), rtol=0.05)

# ╔═╡ f8be3a4b-d8f0-40cb-869e-81c17e58327a
md"###### Consider those tests as bonus"

# ╔═╡ 65a8981d-890b-4590-a913-6890ecf3817d
PlutoTest.@test ≈(150, 150 * mean(add_poisson_noise!(ones((512, 512)), 150)), rtol=0.05)

# ╔═╡ 272c8afb-667f-49c7-af48-2feb1b9d06f7
PlutoTest.@test ≈(√(150), 150 * std(add_poisson_noise!(ones((512, 512)), 150)), rtol=0.05)

# ╔═╡ 59482e49-d784-4303-9487-bf37f6a0462e
md"## 1.3 Hot Pixels

Another issue are hot pixels which result in a pixel with maximum value. This can be due to damaged pixels or some other noise (radioactivity, ...). Often this is called Salt (because of the maximum brightness) noise.
" 

# ╔═╡ 6e050fc2-4db2-4e6e-abd4-2812b47c070f
"""
	add_hot_pixels!(img, probability=0.1; max_value=one(eltype(img)))

Add randomly hot pixels. The probability for each pixel to be hot,
should be specified by `probability`.
`max_value` is a keyword argument which is the value the _hot_ pixel will have.
"""
function add_hot_pixels!(img, probability=0.1; max_value=1)
	# todo
	for ii in eachindex(img)
		img[ii] = rand()<probability ? max_value : img[ii]
	end
	return img
end

# ╔═╡ 22985061-9740-45e9-af56-e0787dadba7b
my_show(add_hot_pixels!(copy(img)))

# ╔═╡ b6c25124-3165-4c85-9b1c-da68bd68b59f
my_show(add_hot_pixels!(copy(img), 0.5; max_value=10))

# ╔═╡ 76e83c0b-e9a8-46bd-8a7e-cac39ccaefff
my_show(add_hot_pixels!(copy(img), 0.9))

# ╔═╡ 42f92a8c-54bb-4033-9ef2-f5ffb78f4f3a
md"### 1.3 Test"

# ╔═╡ a9ad9d52-53c9-44ce-8edc-af8d7bfdc81f
PlutoTest.@test sum(map(x -> x ≈ 2, add_hot_pixels!(copy(img), 0.5, max_value=2))) ≈ length(img) .* 0.5 rtol=0.05

# ╔═╡ 373a3e39-fd8c-4cea-88bb-2b9c3734f14f
md"## 2. Remove Noise From Images"

# ╔═╡ 2c274829-ef33-4f34-9ba0-63d3b8cd7343
md"## 2.1 Remove Noise with Gaussian Blur Kernel

One option to remove noise is, is to blur the image with a Gaussian kernel.
We can convolve a small gaussian (odd sized) Kernel over the array.
For that we are using `ImageFiltering.imfilter` for the convolution.

To create a Gaussian shaped function, checkout `IndexFunArrays.normal`. 
You probably want to normalize the sum of it again
"

# ╔═╡ 52ebebfc-940b-47e2-a059-40ec996f2f98
begin
	img_g = add_gauss_noise(img, 0.1)
	img_p = add_poisson_noise!(100 .* copy(img)) ./ 100
	img_h = add_hot_pixels!(copy(img))
end;

# ╔═╡ 2d6f25a5-b902-4e46-ac3f-cf452f525912
my_show([img_g img_p img_h])

# ╔═╡ 75aa8212-8ba3-4fba-b8b2-af2dd9c82adf
# check out the help
imfilter

# ╔═╡ ded0b153-3ba6-4ad6-906b-87930e2b82d0
# check out the help
normal

# ╔═╡ 6acf7ba6-445d-4ac0-a83f-6916318c783c
function gaussian_noise_remove(arr; kernel_size=(3,3), σ=1)
	# todo
	gaussian_kernel = normal(kernel_size, sigma = σ)
	gaussian_kernel1 = gaussian_kernel ./ sum(gaussian_kernel)  
	arr = imfilter(arr, gaussian_kernel1)
	return arr
	#return similar(arr)
end

# ╔═╡ 4f683117-9a09-4837-b0af-5818ac2e62fd
md"σ = $(@bind σ Slider(0.01:0.1:10, show_value=true))"

# ╔═╡ 01203c2c-4e67-492c-9d9c-dc84a84afe15
md"kernel size = $(@bind ks Slider(3:2:21, show_value=true))"

# ╔═╡ daae9a7d-99f4-4d34-a7fd-886fa754d59b
begin
	img_p_gauss = gaussian_noise_remove(img_p, kernel_size=(ks,ks), σ=σ);
	img_g_gauss = gaussian_noise_remove(img_g, kernel_size=(ks,ks), σ=σ);
	img_h_gauss = gaussian_noise_remove(img_h, kernel_size=(ks,ks), σ=σ);
end;

# ╔═╡ 92d43328-29a2-45b2-ae71-e169dec02316
my_show([img_p_gauss img_g_gauss img_h_gauss])

# ╔═╡ 512486eb-1919-446a-9f24-e9713546d237
md"### 2.1 Test"

# ╔═╡ 9b83d92d-f838-4329-b0b1-62bf7a18f123
arr_rand = add_gauss_noise(ones((500, 500)), 0.2);

# ╔═╡ bbbed1b7-a406-44a7-8c77-959a3eba0197
PlutoTest.@test std(gaussian_noise_remove(arr_rand, kernel_size=(8,8), σ=2)) < 0.05

# ╔═╡ 911c2e1e-f04a-4efa-9aff-7f4a424011ae
PlutoTest.@test sum(abs2, img_g .- img) > sum(abs2, img_g_gauss .- img)

# ╔═╡ 0f6ff33f-e50a-4317-a240-3b2f52de7a97
md"## 2.2 Noise Removal with Median Filter
The median filter slides with a quadratic shaped box over the array and
always takes the median value of this array.

So conceptually, you can use two for loops over the first two dimensions of the array. Then extract a region around this point and calculate the median.
Try to preserve the image output size. Assign the median to this pixel.
"

# ╔═╡ 7bc0f845-d5d4-4192-be1d-b97750e95761
function median_noise_remove!(arr; kernel_size=(5,5))
	# todo
    output = mapwindow(median, arr, kernel_size)
    arr .= output
	#return similar(arr)
	return arr
end

# ╔═╡ ac6e7ea7-c277-4180-a809-a37531ab4f11
md"kernel\_size\_2 = $(@bind ks_2 Slider(3:2:9, show_value=true))"

# ╔═╡ 8208b936-ad5d-45a0-96d1-04fd131f1e29
begin
	img_p_median = median_noise_remove!(copy(img_p), kernel_size=(ks_2,ks_2));
	img_g_median = median_noise_remove!(copy(img_g), kernel_size=(ks_2,ks_2));
	img_h_median = median_noise_remove!(copy(img_h), kernel_size=(ks_2,ks_2));
end;

# ╔═╡ 06af0b94-f4e1-4114-b565-730b87567995
my_show([img_p_median img_g_median img_h_median])

# ╔═╡ 707a3be6-ac83-421d-a8e1-31d75444aeef
md"### 2.2 Test"

# ╔═╡ 07f5e6da-7868-4e22-90aa-7a81c2b41ab3
PlutoTest.@test sum(abs2, img_p .- img) > sum(abs2, img_p_median .- img)

# ╔═╡ 3dbe2b38-8609-41e5-b3b9-d62da394b89d
PlutoTest.@test sum(abs2, img_g .- img) > sum(abs2, img_g_median .- img)

# ╔═╡ 867d5676-8dfc-4bb5-b555-73aa767f7e9e
md"## 3 Final Images"

# ╔═╡ 3605fda7-078e-4bff-9e33-91c4025e775a
my_show([img_g img_p img_h])

# ╔═╡ c09b3470-95bd-4f49-b64b-26459c95ab19
my_show([median_noise_remove!(copy(img_g)) median_noise_remove!(copy(img_p)) median_noise_remove!(copy(img_h))])  

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
IndexFunArrays = "613c443e-d742-454e-bfc6-1d7f8dd76566"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PoissonRandom = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
Colors = "~0.12.8"
ImageFiltering = "~0.7.2"
ImageShow = "~0.3.6"
IndexFunArrays = "~0.2.5"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.44"
PoissonRandom = "~0.4.1"
TestImages = "~1.7.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "70c7280f6f75c29d18c18c0725f7df689ae507ce"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "d80af0733c99ea80575f612813fa6aa71022d33a"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.0"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "3640d077b6dafd64ceb8fd5c1ec76f7ca53bcf76"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.16.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
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
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "13951eb68769ad1cd460cdb2e64e5e95f1bf123d"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.0"

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

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

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

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

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
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "62ca0547a14c57e98154423419d8a342dca75ca9"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.4"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

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
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

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
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

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
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

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

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

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

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

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

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "98ca7c29edd6fc79cd74c61accb7010a4e7aee33"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.6.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

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
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

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

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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
version = "1.11.0"

[[deps.TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "ImageIO", "ImageMagick", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "03492434a1bdde3026288939fc31b5660407b624"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.7.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "38f139cc4abf345dd4f22286ec000728d5e8e097"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "7dfa0fd9c783d3d0cc43ea1af53d69ba45c447df"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

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
# ╟─31cc54e9-d47b-4df7-a785-dad0120e8cdb
# ╠═8a11c2b7-b6e5-4c26-9e06-ad3ab958db9c
# ╠═eb8a1967-7b7a-4ee8-851a-992c9dd13486
# ╠═b731d80c-3012-49f5-8e63-9b7cea0ed667
# ╟─d8c83f2c-76ea-42b9-bd39-7fa5cafcd4d3
# ╠═334ebdc5-dc11-4e83-ad12-a1961420a97b
# ╠═7a7e06df-dba4-4a55-81bc-f7129ab7d874
# ╟─2122c12f-0832-4f19-9a77-3047d5311cba
# ╟─a7d7a415-7464-43fb-a2bf-8e3f2d999c32
# ╠═de104b0a-28a4-42de-ae8d-fa857a8f32e0
# ╟─f89bc1e4-d406-4550-988b-71496b64035a
# ╠═c0b5ff08-7342-4c25-9791-be3a4918c400
# ╟─d56bc500-f5cf-40f3-8c0b-b4ae26a58367
# ╠═1116455e-4698-489a-add2-4e1d8ecbb546
# ╟─cc7bbb55-b8dd-454b-b170-45137ab0c0b5
# ╠═c997c248-fba8-4ed8-ab75-179d248e6b83
# ╠═07bc00b7-81ad-4645-8fbe-8545ccdfe7e4
# ╠═d317ed1a-1867-4293-8db4-51352212929c
# ╟─e69a8b77-833a-4498-8bc9-e4c970f6529d
# ╠═3ccbf4fb-0707-4ce8-83d8-b1af555f3fe1
# ╠═51a7dfc9-bfdc-46da-bba5-dcbee4ba5770
# ╠═1e4b32cb-00ea-439f-900e-abadc850be95
# ╠═6cf8ff2a-4255-4c53-9376-5a0d2d372569
# ╠═9a3feb13-0424-448c-934b-45c476cb0096
# ╟─e3fcd381-9652-4e1d-8585-a91f87b73ce5
# ╠═f3efcfcf-4516-43d2-9375-d8863b4b6f5b
# ╠═5b68030f-557e-449e-8baf-cf7ae65e4425
# ╠═78a39a77-bc9d-427a-a113-4e64b0c9d311
# ╠═41118b3c-3e49-4fc7-8c80-7906b7cf91fe
# ╟─f8be3a4b-d8f0-40cb-869e-81c17e58327a
# ╠═65a8981d-890b-4590-a913-6890ecf3817d
# ╠═272c8afb-667f-49c7-af48-2feb1b9d06f7
# ╟─59482e49-d784-4303-9487-bf37f6a0462e
# ╠═6e050fc2-4db2-4e6e-abd4-2812b47c070f
# ╠═22985061-9740-45e9-af56-e0787dadba7b
# ╠═b6c25124-3165-4c85-9b1c-da68bd68b59f
# ╠═76e83c0b-e9a8-46bd-8a7e-cac39ccaefff
# ╟─42f92a8c-54bb-4033-9ef2-f5ffb78f4f3a
# ╠═a9ad9d52-53c9-44ce-8edc-af8d7bfdc81f
# ╟─373a3e39-fd8c-4cea-88bb-2b9c3734f14f
# ╟─2c274829-ef33-4f34-9ba0-63d3b8cd7343
# ╠═52ebebfc-940b-47e2-a059-40ec996f2f98
# ╠═2d6f25a5-b902-4e46-ac3f-cf452f525912
# ╠═62c6e2f0-0007-40ec-81c5-61cda80f59e5
# ╠═75aa8212-8ba3-4fba-b8b2-af2dd9c82adf
# ╠═ded0b153-3ba6-4ad6-906b-87930e2b82d0
# ╠═6acf7ba6-445d-4ac0-a83f-6916318c783c
# ╟─4f683117-9a09-4837-b0af-5818ac2e62fd
# ╟─01203c2c-4e67-492c-9d9c-dc84a84afe15
# ╠═daae9a7d-99f4-4d34-a7fd-886fa754d59b
# ╠═92d43328-29a2-45b2-ae71-e169dec02316
# ╟─512486eb-1919-446a-9f24-e9713546d237
# ╠═9b83d92d-f838-4329-b0b1-62bf7a18f123
# ╠═bbbed1b7-a406-44a7-8c77-959a3eba0197
# ╠═911c2e1e-f04a-4efa-9aff-7f4a424011ae
# ╟─0f6ff33f-e50a-4317-a240-3b2f52de7a97
# ╠═7bc0f845-d5d4-4192-be1d-b97750e95761
# ╟─ac6e7ea7-c277-4180-a809-a37531ab4f11
# ╠═8208b936-ad5d-45a0-96d1-04fd131f1e29
# ╠═06af0b94-f4e1-4114-b565-730b87567995
# ╟─707a3be6-ac83-421d-a8e1-31d75444aeef
# ╠═07f5e6da-7868-4e22-90aa-7a81c2b41ab3
# ╠═3dbe2b38-8609-41e5-b3b9-d62da394b89d
# ╟─867d5676-8dfc-4bb5-b555-73aa767f7e9e
# ╠═3605fda7-078e-4bff-9e33-91c4025e775a
# ╠═c09b3470-95bd-4f49-b64b-26459c95ab19
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

import replicate

output = replicate.run(
    "zsxkib/flux-dev-inpainting:ca8350ff748d56b3ebbd5a12bd3436c2214262a4ff8619de9890ecc41751a008",
    input={
        "mask": "https://replicate.delivery/pbxt/HtGQBqO9MtVbPm0G0K43nsvvjBB0E0PaWOhuNRrRBBT4ttbf/mask.png",
        "image": "https://replicate.delivery/pbxt/HtGQBfA5TrqFYZBf0UL18NTqHrzt8UiSIsAkUuMHtjvFDO6p/overture-creations-5sI6fQgYIuo.png",
        "width": 1024,
        "height": 1024,
        "prompt": "small cute cat sat on a park bench",
        "strength": 1,
        "num_outputs": 1,
        "output_format": "webp",
        "guidance_scale": 7,
        "output_quality": 90,
        "num_inference_steps": 30
    }
)

# The zsxkib/flux-dev-inpainting model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
for item in output:
    # https://replicate.com/zsxkib/flux-dev-inpainting/api#output-schema
    print(item)
    #=> "https://replicate.delivery/yhqm/M3kYRwlZQ0YiG1PJaeIsDEnsO1JJozDQVbrVqMNjqETLEopJA/output.webp"
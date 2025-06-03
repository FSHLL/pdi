
import SimpleITK as sitk
import numpy as np

def apply_registration(image, second_image):
    fixed_np = image.astype(np.float32)
    moving_np = second_image.astype(np.float32)

    fixed_sitk = sitk.GetImageFromArray(fixed_np)
    moving_sitk = sitk.GetImageFromArray(moving_np)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_sitk,
        moving_sitk,
        sitk.Similarity3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

    registered_sitk = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_sitk.GetPixelID()
    )

    return sitk.GetArrayFromImage(registered_sitk)
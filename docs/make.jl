using Documenter
using WarningDissem

DocMeta.setdocmeta!(WarningDissem, :DocTestSetup, :(using WarningDissem); recursive=true)

makedocs(
    sitename = "WarningDissem",
    format = Documenter.HTML(),
    modules = [WarningDissem]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

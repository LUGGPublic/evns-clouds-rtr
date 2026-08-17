add_mandrill_executable(Clouds)

target_sources(Clouds
	PRIVATE
		"MLP.cpp"
		"NeuralMesh.cpp"
		"Clouds.cpp"
)

add_shaders(Clouds
	"Cloud.frag"
	"Cloud.vert"
	"Depth.frag"
	"Fullscreen.vert"
	"Position.frag"
	"Resolve.frag"
	"Shadow.frag"
	"Sky.frag"
)

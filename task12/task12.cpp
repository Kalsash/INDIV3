#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SOIL/SOIL.h> 
#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
using namespace std;

struct Texture {
    unsigned int id;
    string type;
    string path;  // we store the path of the texture to compare with other textures
};

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

class Mesh {
public:
    // mesh data
    vector<Vertex> vertices;
    vector<unsigned int> indices;
    vector<Texture> textures;
    unsigned int VAO, VBO, EBO;
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures)
    {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;

        // now that we have all the required data, set the vertex buffers and its attribute pointers.
        setupMesh();
    }
    // render the mesh
    void Draw(GLuint shader_id, GLint count)
    {
        //связываем текстуры
        unsigned int diffuseNr = 1;
        unsigned int specularNr = 1;
        unsigned int normalNr = 1;
        unsigned int heightNr = 1;

        for (unsigned int i = 0; i < textures.size(); i++)
        {
            //активируем нужную текстуру
            glActiveTexture(GL_TEXTURE0 + i);

            // так как мы не знаем количество и тип структур, 
            // то будем по номеру каждой входной текстуры и ее типу добавлять к соответственой переменной номер текущей текстуры 
            std::string number;
            std::string name = textures[i].type;
            if (name == "texture_diffuse")
                number = std::to_string(diffuseNr++);
            else if (name == "texture_specular")
                number = std::to_string(specularNr++);
            else if (name == "texture_normal")
                number = std::to_string(normalNr++);
            else if (name == "texture_height")
                number = std::to_string(heightNr++);

            // передаем в шейдер
            glUniform1i(glGetUniformLocation(shader_id, (name + number).c_str()), i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }

        glBindVertexArray(VAO);
        glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0, count);
        glBindVertexArray(0);

        // отключаем текстуру
        glActiveTexture(GL_TEXTURE0);
    }
private:

    // initializes all the buffer objects/arrays
    void setupMesh()
    {
        // create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        // load data into vertex buffers
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        // A great thing about structs is that their memory layout is sequential for all its items.
        // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
        // again translates to 3/2 floats which translates to a byte array.
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // set the vertex attribute pointers
        // vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // vertex texture coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

        glBindVertexArray(0);
    }
};
unsigned int TextureFromFile(const char* path, const string& directory, bool gamma = false);
class Model
{
public:
    // model data 
    vector<Texture> textures_loaded;	// stores all the textures loaded so far, optimization to make sure textures aren't loaded more than once.
    vector<Mesh> meshes;
    string directory;
    bool gammaCorrection;

    // constructor, expects a filepath to a 3D model.
    Model(string const& path, bool gamma = false) : gammaCorrection(gamma)
    {
        loadModel(path);
    }

    // draws the model, and thus all its meshes
    void Draw(GLuint shader, GLint count)
    {
        for (unsigned int i = 0; i < meshes.size(); i++)
            meshes[i].Draw(shader, count);
    }

private:

    void loadModel(string path)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
            return;
        }
        directory = path.substr(0, path.find_last_of('/'));

        processNode(scene->mRootNode, scene);
    }
    void processNode(aiNode* node, const aiScene* scene)
    {
        // process all the node's meshes (if any)
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }
        // then do the same for each of its children
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene);
        }
    }
    Mesh processMesh(aiMesh* mesh, const aiScene* scene)
    {
        // data to fill
        vector<Vertex> vertices;
        vector<unsigned int> indices;
        vector<Texture> textures;

        // walk through each of the mesh's vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;
            glm::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
            // positions
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.Position = vector;
            // normals
            if (mesh->HasNormals())
            {
                vector.x = mesh->mNormals[i].x;
                vector.y = mesh->mNormals[i].y;
                vector.z = mesh->mNormals[i].z;
                vertex.Normal = vector;
            }
            // texture coordinates
            if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
            {
                glm::vec2 vec;
                // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
                // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
                vec.x = mesh->mTextureCoords[0][i].x;
                vec.y = mesh->mTextureCoords[0][i].y;
                vertex.TexCoords = vec;
            }
            else
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);

            vertices.push_back(vertex);
        }
        // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            // retrieve all indices of the face and store them in the indices vector
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }
        // process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        // 1. diffuse maps
        vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
        // 2. specular maps
        vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
        // 3. normal maps
        std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
        // 4. height maps
        std::vector<Texture> heightMaps = loadMaterialTextures(material, aiTextureType_AMBIENT, "texture_height");
        textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());

        // return a mesh object created from the extracted mesh data
        return Mesh(vertices, indices, textures);
    }
    vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, string typeName)
    {
        vector<Texture> textures;
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
        {
            aiString str;
            mat->GetTexture(type, i, &str);
            // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
            bool skip = false;
            for (unsigned int j = 0; j < textures_loaded.size(); j++)
            {
                if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
                {
                    textures.push_back(textures_loaded[j]);
                    skip = true; // a texture with the same filepath has already been loaded, continue to next one. (optimization)
                    break;
                }
            }
            if (!skip)
            {   // if texture hasn't been loaded already, load it
                Texture texture;
                texture.id = TextureFromFile(str.C_Str(), this->directory);
                texture.type = typeName;
                texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);  // store it as texture loaded for entire model, to ensure we won't unnecessary load duplicate textures.
            }
        }
        return textures;
    }
};
unsigned int TextureFromFile(const char* path, const string& directory, bool gamma)
{
    string filename = string(path);
    filename = directory + '/' + filename;

    unsigned int textureID;
    glGenTextures(1, &textureID);

    // грузим картинку
    int width, height;
    int nrComponents = 3;
    unsigned char* data = SOIL_load_image(filename.c_str(), &width, &height, 0, nrComponents);

    if (data)
    {
        GLenum format = GL_RGB;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        SOIL_free_image_data(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        SOIL_free_image_data(data);
    }

    return textureID;
}

enum axis {
    OX,
    OY,
    OZ,
    NUL
};

axis ax = NUL;

enum modeModel {
    tree,
    planet1,
    planet2,
    simple
};


int sectors = 0;
bool two_tex = false;

float mixValue = 0.4f;


//camera
glm::vec3 cameraPos = glm::vec3(0.0f, 4.0f, 35.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// время для обработки кадров
float deltaTime = 0.0f;
float lastFrame = 0.0f;

//вращение
bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float last_x = 450.0f;
float last_y = 450.0f;

// ID шейдерной программы
GLuint Tree_mode;
GLuint Box_mode;

// ID атрибута
GLint Attrib_vertex;

// ID Vertex Buffer Object
GLuint VBO_planet1;
GLuint VBO_planet2;

//инстансинг
GLuint quantity = 2
;
glm::mat4* modelMatrices;
glm::mat4* modelMatricesToPosition;
glm::mat4* modelMatricesToCenter;
glm::mat4* localRotateMatrices;
glm::mat4* tmpModelMatrices;




// Исходный код вершинного шейдера

const char* VertexShaderTree = R"(
    #version 330 core

    layout (location = 0) in vec3 coord_pos;
    layout (location = 2) in vec2 tex_coord_in;

    out vec2 coord_tex;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main() 
    { 
        gl_Position = projection * view * model * vec4(coord_pos, 1.0);
        coord_tex = tex_coord_in;
        //coord_tex = vec2(tex_coord_in.x, 1.0f - tex_coord_in.y); //если текстуры неправильно наложились
    }
    )";

const char* VertexShaderPlanet = R"(
    #version 330 core

    layout (location = 0) in vec3 coord_pos;
    layout (location = 2) in vec2 tex_coord_in;

    out vec2 coord_tex;

    uniform mat4 instanceModel[15];
    uniform mat4 view;
    uniform mat4 projection;
    
    void main() 
    { 
        gl_Position = projection * view * instanceModel[gl_InstanceID] * vec4(coord_pos, 1.0);
        coord_tex = tex_coord_in;
        //coord_tex = vec2(tex_coord_in.x, 1.0f - tex_coord_in.y); //если текстуры неправильно наложились
    }
    )";

// Исходный код фрагментного шейдера

const char* FragShaderTree = R"(
    #version 330 core
    
    in vec2 coord_tex;

    uniform sampler2D texture_diffuse1;

    void main()  
    {
       gl_FragColor = texture(texture_diffuse1, coord_tex);
    } 
)";

void checkOpenGLerror()
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::cout << "OpenGL error " << err << std::endl;
    }
}

void ShaderLog(unsigned int shader)
{
    int infologLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLen);
    GLint vertex_compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &vertex_compiled);

    if (infologLen > 1)
    {
        int charsWritten = 0;
        std::vector<char> infoLog(infologLen);
        glGetShaderInfoLog(shader, infologLen, &charsWritten, infoLog.data());
        std::cout << "InfoLog: " << infoLog.data() << std::endl;
    }

    if (vertex_compiled != GL_TRUE)
    {
        GLsizei log_length = 0;
        GLchar message[1024];
        glGetShaderInfoLog(shader, 1024, &log_length, message);
        std::cout << "InfoLog2: " << message << std::endl;
    }

}
void InitShaderTree() {
    GLuint vShaderTree = glCreateShader(GL_VERTEX_SHADER);
    //компиляция шейдера
    glShaderSource(vShaderTree, 1, &VertexShaderTree, NULL);
    glCompileShader(vShaderTree);
    std::cout << "vertex shader \n";
    ShaderLog(vShaderTree);

    GLuint fShaderTree = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fShaderTree, 1, &FragShaderTree, NULL);
    glCompileShader(fShaderTree);
    std::cout << "fragment shader \n";
    // Функция печати лога шейдера
    ShaderLog(fShaderTree);
    Tree_mode = glCreateProgram();
    glAttachShader(Tree_mode, vShaderTree);
    glAttachShader(Tree_mode, fShaderTree);
    // Линкуем шейдерную программу
    glLinkProgram(Tree_mode);
    // Проверяем статус сборки
    int link_ok;
    glGetProgramiv(Tree_mode, GL_LINK_STATUS, &link_ok);

    if (!link_ok)
    {
        std::cout << "error attach shaders \n";
        return;
    }
    checkOpenGLerror();
}

void InitShaderBox()
{
    GLuint vShaderBox = glCreateShader(GL_VERTEX_SHADER);
    // Передаем исходный код
    glShaderSource(vShaderBox, 1, &VertexShaderPlanet, NULL);
    // Компилируем шейдер
    glCompileShader(vShaderBox);
    std::cout << "vertex shader t\n";
    // Функция печати лога шейдера
    ShaderLog(vShaderBox);

    // Создаем фрагментный шейдер
    GLuint fShaderBox = glCreateShader(GL_FRAGMENT_SHADER);
    // Передаем исходный код
    glShaderSource(fShaderBox, 1, &FragShaderTree, NULL);
    // Компилируем шейдер
    glCompileShader(fShaderBox);
    std::cout << "fragment shader \n";
    // Функция печати лога шейдера
    ShaderLog(fShaderBox);
    Box_mode = glCreateProgram();
    glAttachShader(Box_mode, vShaderBox);
    glAttachShader(Box_mode, fShaderBox);

    // Линкуем шейдерную программу
    glLinkProgram(Box_mode);
    int link_ok;
    // Проверяем статус сборки
    glGetProgramiv(Box_mode, GL_LINK_STATUS, &link_ok);

    if (!link_ok)
    {
        std::cout << "error attach shaders \n";
        return;
    }
    checkOpenGLerror();
}
void InitShader()
{

    InitShaderTree();
    InitShaderBox();

}


void Init()
{
    // Шейдеры
    InitShader();


    //включаем тест глубины
    glEnable(GL_DEPTH_TEST);
}


void Draw(sf::Clock clock, Model mod, modeModel mode, int count)
{
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 900.0f / 900.0f, 0.1f, 100.0f);
    switch (mode)
    {
    case (tree):
    {


        glUseProgram(Tree_mode); // Устанавливаем шейдерную программу текущей

        float angle = 25.0f;

        model = glm::scale(model, glm::vec3(2.5f, 2.5f, 2.5f));
        model = glm::translate(model, glm::vec3(0.0f, 0, 0.0f));

        model = glm::rotate(model, clock.getElapsedTime().asSeconds() * glm::radians(angle), glm::vec3(0.0f, 1.0f, 0.0f));
        view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        projection = glm::perspective(glm::radians(45.0f), 900.0f / 900.0f, 0.1f, 100.0f);

        glUniformMatrix4fv(glGetUniformLocation(Tree_mode, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(Tree_mode, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(Tree_mode, "model"), 1, GL_FALSE, glm::value_ptr(model));

        mod.Draw(Tree_mode, count);
        glUseProgram(0); // Отключаем шейдерную программу
    }
    break;
    case (simple):
    {


        glUseProgram(Tree_mode); // Устанавливаем шейдерную программу текущей

        float angle = -90.0f;

        model = glm::scale(model, glm::vec3(0.1f, 0.1f, 0.1f));
        model = glm::translate(model, glm::vec3(7.0f, 0, 0.0f));

        model = glm::rotate(model,  glm::radians(angle), glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        projection = glm::perspective(glm::radians(45.0f), 900.0f / 900.0f, 0.1f, 100.0f);

        glUniformMatrix4fv(glGetUniformLocation(Tree_mode, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(Tree_mode, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(Tree_mode, "model"), 1, GL_FALSE, glm::value_ptr(model));

        mod.Draw(Tree_mode, count);
        glUseProgram(0); // Отключаем шейдерную программу
    }
    break;
    case (planet1):
    {
        glUseProgram(Box_mode); // Устанавливаем шейдерную программу текущей
        float angle = 25.0f;
        float size = 200;
        for (int i = 0; i < quantity; i++)
        {
            model = glm::rotate(glm::mat4(1.0f), clock.getElapsedTime().asSeconds() * glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            model = glm::scale(model, glm::vec3(size, size, size));
            size += 50;
            model = glm::translate(model, glm::vec3(15.0f, 5, 5.0f));
            tmpModelMatrices[i] = modelMatrices[i] * modelMatricesToCenter[i] * localRotateMatrices[i] * glm::rotate(glm::mat4(1.0f), clock.getElapsedTime().asSeconds() * glm::radians(25.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * modelMatricesToPosition[i] * model;
        }
        view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);


        glUniformMatrix4fv(glGetUniformLocation(Box_mode, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(Box_mode, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(Box_mode, "instanceModel"), quantity, GL_FALSE, &tmpModelMatrices[0][0][0]);

        mod.Draw(Box_mode, count);

        glUseProgram(0); // Отключаем шейдерную программу

    }
    break;

    checkOpenGLerror();
    }
}


// Освобождение буфера
void ReleaseVBO()
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &VBO_planet1);
    glDeleteBuffers(1, &VBO_planet2);
}

// Освобождение шейдеров
void ReleaseShader()
{
    // Передавая ноль, мы отключаем шейдерную программу
    glUseProgram(0);
    // Удаляем шейдерные программы
    glDeleteProgram(Tree_mode);
    glDeleteProgram(Box_mode);
}


void Release()
{
    // Шейдеры
    ReleaseShader();
    // Вершинный буфер
    ReleaseVBO();
}



//https://russianblogs.com/article/455565970/
void runner() {
    std::setlocale(LC_ALL, "Russian");
    sf::Window window(sf::VideoMode(1000, 1000), "Lab13", sf::Style::Default, sf::ContextSettings(24));
    window.setVerticalSyncEnabled(true);
    window.setActive(true);
    glClearColor(0.529, 0.808, 0.922, 2.0);

    glewInit(); 
    glGetError(); // сброс флага GL_INVALID_ENUM

    Init();

    modelMatrices = new glm::mat4[quantity];
    localRotateMatrices = new glm::mat4[quantity];
    modelMatricesToPosition = new glm::mat4[quantity];
    modelMatricesToCenter = new glm::mat4[quantity];
    tmpModelMatrices = new glm::mat4[quantity];
    sf::Clock clock;
    srand(static_cast<unsigned int>(clock.getElapsedTime().asSeconds())); // initialize random seed

    Model centralModel("tree/source/Christmas_tree.obj");
    Model planet1_model("planet1/penguin02.fbx");
    Model field_model("kek/10450_Rectangular_Grass_Patch_v1_iterations-2.obj");


    //Model centralModel3("planet1/penguin02.fbx");
    float radius = 10.0;
    float offset = 4.0f;
    for (unsigned int i = 0; i < quantity; i++)
    {
        glm::mat4 model = glm::mat4(1.0f);
        // перемещаем по осям
        float angle = (float)i / (float)quantity * 360.0f;

        float displacement = (rand() % (int)(offset * 100)) / 100.0f;
        float x = sin(angle) * radius + displacement;

        displacement = (rand() % (int)(offset * 100)) / 100.0f - offset;
        float y = (displacement + 10.0f);

        float z = cos(angle) * radius + displacement;

        modelMatricesToPosition[i] = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z) * 10.0f);
        modelMatricesToCenter[i] = glm::translate(glm::mat4(1.0f), glm::vec3(-x, -y, -z) * 10.0f);
        model = glm::translate(model, glm::vec3(0.0f, 3.0f, 0.0f));

        // размер
        float scale = static_cast<float>((rand() % 40) / 100000.0 + 0.0015);
        model = glm::scale(model, glm::vec3(scale));

        //поворот
        float rotAngle = static_cast<float>((rand() % 360));

        localRotateMatrices[i] = glm::rotate(glm::mat4(1.0f), (float)(rand() % 20), glm::vec3(1.0f, 5.0f, 0.0f));

        modelMatrices[i] = model * modelMatricesToPosition[i];
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            float currentFrame = static_cast<float>(clock.getElapsedTime().asSeconds());
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
            float cameraSpeed = 0.5f;
            if (event.type == sf::Event::Closed)
                window.close();
            else if (event.type == sf::Event::Resized)
                glViewport(0, 0, event.size.width, event.size.height);
            else if (event.type == sf::Event::KeyPressed)
            {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                    window.close();
                    break;
                case sf::Keyboard::W:
                    cameraPos += cameraSpeed * cameraFront;
                    break;
                case sf::Keyboard::S:
                    cameraPos -= cameraSpeed * cameraFront;
                    break;
                case sf::Keyboard::A:
                    cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
                    break;
                case sf::Keyboard::D:
                    cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
                    break;

                default:
                    break;
                }
            }
            if (event.type == sf::Event::MouseMoved)
            {
                float xpos = static_cast<float>(event.mouseMove.x);
                float ypos = static_cast<float>(event.mouseMove.y);

                if (firstMouse)
                {
                    last_x = xpos;
                    last_y = ypos;
                    firstMouse = false;
                }

                float xoffset = xpos - last_x;
                float yoffset = last_y - ypos;
                last_x = xpos;
                last_y = ypos;

                float sensitivity = 0.1f;
                xoffset *= sensitivity;
                yoffset *= sensitivity;

                yaw += xoffset;
                pitch += yoffset;

                if (pitch > 89.0f)
                    pitch = 89.0f;
                if (pitch < -89.0f)
                    pitch = -89.0f;

                glm::vec3 front;
                front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
                front.y = sin(glm::radians(pitch));
                front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
                cameraFront = glm::normalize(front);
            }

        }
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

       Draw(clock, centralModel, tree, 1);
        Draw(clock, planet1_model, planet1, quantity);
        Draw(clock, field_model, simple, 1);
        window.display();
    }
}

int main()
{
    runner();
    Release();
    return 0;
}


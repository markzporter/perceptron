using CSV, Plots

# Perceptron() takes X, y, w, b as defined in the perceptron
# algorithm. It also takes max pass which is the number of times
# it will run the perceptron algorithm on the datase. 
# Returns a 3-tuple (w, b, mistake) of the trained model. 
function perceptron(X, y, w, b, max_pass)
    n = size(X)[2]
    mistake = zeros(Int, max_pass)
    padding = ones(Int, n)
    A = [X; padding']'
    w = [w; b]
    for t in 1:max_pass
        for i in 1:n
            a = y[i] * A[i, :]
            if a'w <= 0
                w = w + a
                mistake[t] = mistake[t] + 1
            end
        end
    end
    return (w[1:end-1], w[end], mistake)
end 


# Fetch X and y data 
X = convert(Array, CSV.read("./spambase_X.csv", header=false))
y = convert(Array, CSV.read("./spambase_Y.csv", header=false))

# Number of passes to train the model
max_pass = 500


d = size(X)[1]
w = zeros(Int, d)
b = 0

result = perceptron(X, y, w, b, max_pass)

mistake = result[3]
p = plot(1:max_pass, mistake, title="Mistakes made during model training")
xlabel!("Passes")
ylabel!("Mistakes")
savefig("myplot.png") # Saves the CURRENT_PLOT as a .png
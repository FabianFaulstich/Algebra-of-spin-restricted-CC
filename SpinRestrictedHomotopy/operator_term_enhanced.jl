# Enhanced display for FD operator types
using HomotopyContinuation

# Helper function to check if an expression needs parentheses
function needs_parentheses(expr::Union{Expression, Variable, Number})
    # Only add parentheses for Expressions that are sums with multiple terms
    if typeof(expr) == Expression
        expr_str = string(expr)
        # Check if it's a sum/difference with multiple terms (not just a negative number)
        has_plus = occursin("+", expr_str)
        has_minus_not_at_start = occursin("-", expr_str[2:end])
        return has_plus || has_minus_not_at_start
    end
    return false
end

# Helper function to format expressions nicely
function format_expression(expr::Union{Expression, Variable, Number})
    expr_str = string(typeof(expr) == Expression ? expand(expr) : expr)
    # Clean up variable formatting if needed
    return expr_str
end

# Override the plain text display to use LaTeX format
function Base.show(io::IO, op::Op)
    if op.creation
        print(io, "a_{$(op.index)}^†")
    else
        print(io, "a_{$(op.index)}")
    end
end

# Enhanced display for FDword type
function Base.show(io::IO, word::FDword)
    if isnothing(word.monomial)
        # Just a constant
        print(io, format_expression(word.constant))
    else
        # Handle coefficient
        if word.constant == 1
            # Don't print coefficient
        elseif word.constant == -1
            print(io, "-")
        else
            const_str = format_expression(word.constant)
            if needs_parentheses(word.constant)
                print(io, "(", const_str, ")")
            else
                print(io, const_str)
            end
        end
        
        # Print operators
        for op in word.monomial
            if op.creation
                print(io, "a_{$(op.index)}^†")
            else
                print(io, "a_{$(op.index)}")
            end
        end
    end
end

# Enhanced display for FD type (sum of FDwords)
function Base.show(io::IO, fd::FD)
    if isempty(fd.sum)
        print(io, "0")
        return
    end
    
    for (i, word) in enumerate(fd.sum)
        if i > 1
            # Check if the constant is negative to avoid double signs
            if word.constant isa Expression || word.constant isa Variable
                const_str = string(expand(word.constant))
                if !needs_parentheses(word.constant) && startswith(const_str, "-")
                    print(io, " ")
                else
                    print(io, " + ")
                end
            elseif word.constant isa Number && word.constant < 0
                print(io, " ")
            else
                print(io, " + ")
            end
        end
        
        # Display the word
        if isnothing(word.monomial)
            # Just a constant
            print(io, format_expression(word.constant))
        else
            # Handle coefficient
            if word.constant == 1
                # Don't print coefficient
            elseif word.constant == -1
                print(io, "-")
            else
                const_str = format_expression(word.constant)
                if needs_parentheses(word.constant)
                    print(io, "(", const_str, ")")
                else
                    print(io, const_str)
                end
            end
            
            # Print operators
            for op in word.monomial
                if op.creation
                    print(io, "a_{$(op.index)}^†")
                else
                    print(io, "a_{$(op.index)}")
                end
            end
        end
    end
end

# Jupyter notebook LaTeX display methods
function Base.show(io::IO, ::MIME"text/latex", op::Op)
    if op.creation
        print(io, "\$a_{$(op.index)}^{\\dagger}\$")
    else
        print(io, "\$a_{$(op.index)}\$")
    end
end

function Base.show(io::IO, ::MIME"text/latex", word::FDword)
    if isnothing(word.monomial)
        # Just a constant
        print(io, "\$", format_expression(word.constant), "\$")
    else
        print(io, "\$")
        
        # Handle coefficient
        if word.constant == 1
            # Don't print coefficient
        elseif word.constant == -1
            print(io, "-")
        else
            const_str = format_expression(word.constant)
            if needs_parentheses(word.constant)
                print(io, "(", const_str, ")")
            else
                print(io, const_str)
            end
        end
        
        # Print operators
        for op in word.monomial
            if op.creation
                print(io, "a_{$(op.index)}^{\\dagger}")
            else
                print(io, "a_{$(op.index)}")
            end
        end
        
        print(io, "\$")
    end
end

function Base.show(io::IO, ::MIME"text/latex", fd::FD)
    if isempty(fd.sum)
        print(io, "\$0\$")
        return
    end
    
    print(io, "\$")
    
    for (i, word) in enumerate(fd.sum)
        if i > 1
            # Check if the constant is negative to avoid double signs
            if word.constant isa Expression || word.constant isa Variable
                const_str = string(expand(word.constant))
                if !needs_parentheses(word.constant) && startswith(const_str, "-")
                    print(io, " ")
                else
                    print(io, " + ")
                end
            elseif word.constant isa Number && word.constant < 0
                print(io, " ")
            else
                print(io, " + ")
            end
        end
        
        # Display the word without the dollar signs (we're already in math mode)
        if isnothing(word.monomial)
            # Just a constant
            print(io, format_expression(word.constant))
        else
            # Handle coefficient
            if word.constant == 1
                # Don't print coefficient
            elseif word.constant == -1
                print(io, "-")
            else
                const_str = format_expression(word.constant)
                if needs_parentheses(word.constant)
                    print(io, "(", const_str, ")")
                else
                    print(io, const_str)
                end
            end
            
            # Print operators
            for op in word.monomial
                if op.creation
                    print(io, "a_{$(op.index)}^{\\dagger}")
                else
                    print(io, "a_{$(op.index)}")
                end
            end
        end
    end
    
    print(io, "\$")
end

# Keep plain text display for terminal
function Base.show(io::IO, m::MIME"text/plain", op::Op)
    if op.creation
        print(io, "a_{$(op.index)}^†")
    else
        print(io, "a_{$(op.index)}")
    end
end

function Base.show(io::IO, m::MIME"text/plain", word::FDword)
    show(io, word)
end

function Base.show(io::IO, m::MIME"text/plain", fd::FD)
    show(io, fd)
end

# Export helper functions if needed
export needs_parentheses, format_expression
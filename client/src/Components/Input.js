import React from "react";

function Input({id, children, className, placeholder, required, type, key}) {
    return <input
        id={id}
        className={className}
        placeholder={placeholder}
        required={required}
        type={type}
        key={key}>
        {children}
    </input>
}

export default Input
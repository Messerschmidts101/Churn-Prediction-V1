import React from "react";

function Input({id, name, children, onChange, className, placeholder, required, type, key}) {
    return <input
        id={id}
        name={name}
        className={className}
        placeholder={placeholder}
        onChange={onChange}
        required={required}
        type={type}
        key={key}>
        {children}
    </input>
}

export default Input
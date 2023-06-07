import React from "react";

function Input({id, children, className, placeholder, type, key}) {
    return <input
        id={id}
        className={className}
        placeholder={placeholder}
        type={type}
        key={key}>
        {children}
    </input>
}

export default Input
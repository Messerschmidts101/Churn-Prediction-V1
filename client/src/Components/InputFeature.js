import React from "react";

function InputFeature({id, feature, children, className, placeholder, required, keyLabel}) {
    return <input
        id={id}
        name={feature}
        className={className}
        placeholder={placeholder}
        required={required}
        type="number"
        step={0.000000001}
        key={keyLabel}>
        {children}
    </input>
}

export default InputFeature
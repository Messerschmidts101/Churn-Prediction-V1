import React from "react";

function InputFeature({children, className, feature, required, key}) {
    return <input
        id={feature}
        className={className}
        feature={feature}
        required={required}
        type="number"
        step={0.000000001}
        key={key}>
        {children}
    </input>
}

export default InputFeature
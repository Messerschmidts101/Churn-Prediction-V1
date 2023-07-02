import React from 'react'
import Header3 from './Header3'
import ScrollButton from './ScrollButton'

function Heroe({display, children, subheading, button, boolSubheading, boolButton}) {
    return (
        <div className="p-5 text-center bg-image rounded-3 bg-image w-100">
            <div className='mask'>
                <div className="d-flex card-title justify-content-center align-items-center h-100">
                    <div className="dark-green">
                        <Header3 className={"mb-3 stroke " + display}>{children}</Header3>
                        {
                            boolSubheading ? <h4 className="mb-3">{subheading}</h4> : ""
                        }
                        {
                            boolButton ? <ScrollButton className="btn-lg" theme={"outline-danger"} href={"#pasteData"}>{button}</ScrollButton> : ""
                        }
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Heroe